#include "ddpm_latent_diffusion.h"
#include "autoencoder.h"
#include "distributions.h"
#include "ddim_sampler.h"

#define MODEL_FILE_EXT ".xml"

void DDPMLatentDiffusion::_init_vae()
{
   auto& core = _config->core;

   auto modelpath = FullPath(_config->model_folder, "vae_feature_extract" MODEL_FILE_EXT);
   std::shared_ptr<ov::Model> model = core.read_model(modelpath);
   logBasicModelInfo(model);
   auto compiledModel = core.compile_model(model, _config->vae_feature_extract_device);
   _vae_feature_extract_infer = compiledModel.create_infer_request();
}

void DDPMLatentDiffusion::_init_ddpm()
{
   auto& core = _config->core;

   std::string subfolder = _config->model_selection == AudioSRModel::BASIC ? "basic" : "speech";
   auto model_folder = FullPath(_config->model_folder, subfolder);
   auto modelpath = FullPath(model_folder, "ddpm" MODEL_FILE_EXT);

   std::cout << "Reading DDPM Model: " << modelpath << std::endl;
   std::shared_ptr<ov::Model> model = core.read_model(modelpath);
   logBasicModelInfo(model);
   auto compiledModel = core.compile_model(model, _config->ddpm__device);
   _ddpm_infer = compiledModel.create_infer_request();
}

void DDPMLatentDiffusion::_init_vocoder()
{
   auto& core = _config->core;

   auto modelpath = FullPath(_config->model_folder, "vocoder" MODEL_FILE_EXT);
   std::shared_ptr<ov::Model> model = core.read_model(modelpath);
   logBasicModelInfo(model);
   auto compiledModel = core.compile_model(model, _config->vocoder_device);
   _vocoder_infer = compiledModel.create_infer_request();
}


DDPMLatentDiffusion::DDPMLatentDiffusion(std::shared_ptr< AudioSR_Config > config)
    : _config(config)
{
    if (!_config)
        throw std::runtime_error("DDPMLatentDiffusion: config is null!");

    _first_stage_encoder = std::make_shared< AutoEncoder >(config);
    _init_vae();
    _sampler = std::make_shared< DDIMSampler >(this);
    _init_ddpm();
    _init_vocoder();
}

void DDPMLatentDiffusion::set_config(AudioSR_Config config)
{
   if (!_config)
      throw std::runtime_error("DDPMLatentDiffusion::set_config: config is null!");

   _first_stage_encoder->set_config(config);

   if (_config->vae_feature_extract_device != config.vae_feature_extract_device)
   {
      _config->vae_feature_extract_device = config.vae_feature_extract_device;
      _init_vae();
   }

   if (_config->ddpm__device != config.ddpm__device ||
      _config->model_selection != config.model_selection)
   {
      _config->ddpm__device = config.ddpm__device;
      _config->model_selection = config.model_selection;
      _init_ddpm();
   }

   if (_config->vocoder_device != config.vocoder_device)
   {
      _config->vocoder_device = config.vocoder_device;
      _init_vocoder();
   }
}

std::pair<torch::Tensor, torch::Tensor> DDPMLatentDiffusion::_get_input(Batch& batch, torch::Generator &gen)
{
    //in LatentDiffusion get_input() first does x = super().get_input(batch, "fbank").
    // and super (DDPM) get_input does:
    // waveform, stft, fbank = (
    // batch["waveform"],
    //    batch["stft"],
    //    batch["log_mel_spec"],
    //    )
    // ...
    // ret["fbank"] = (
    //  fbank.unsqueeze(1).to(memory_format = torch.contiguous_format).float()
    //   )
    // ...
    // return ret["fbank"]
    // And so, it essentially returns: batch["log_mel_spec"].unsqueeze(1).to(memory_format = torch.contiguous_format).float()
    auto x = batch.log_mel_spec.unsqueeze(1).to(torch::kFloat).contiguous();

    auto z = _get_first_stage_encoding(x, gen);

    // xc = super().get_input(batch, "lowpass_mel")
    auto xc = batch.lowpass_mel;

    torch::Tensor c;
    {
        xc = xc.contiguous();
        //dump_tensor(xc, "xc_ov.raw");

        auto xc_ov = wrap_torch_tensor_as_ov(xc);

        _vae_feature_extract_infer.set_input_tensor(xc_ov);
        _vae_feature_extract_infer.infer();
        c = wrap_ov_tensor_as_torch(_vae_feature_extract_infer.get_output_tensor()).clone();
    }

    return { z, c };
}

// Function to find the cutoff index in a tensor
static int64_t _find_cutoff_np(const torch::Tensor& x, double threshold = 0.95) {
    double cutoff_value = x[-1].item<double>() * threshold;
    for (int64_t i = 1; i < x.size(0); ++i) {
        if (x[-i].item<double>() < cutoff_value) {
            return x.size(0) - i;
        }
    }
    return 0;
}


// Function to get the cutoff index using STFT
static int64_t _get_cutoff_index_np(const torch::Tensor& x, int64_t n_fft = 2048, int64_t hop_length = -1, int64_t win_length = -1, bool center = true) {
    // Create a Hann window (default in librosa)
    torch::Tensor window = torch::hann_window(win_length, torch::kFloat32);


    // Perform STFT on the input signal (similar to librosa.stft)
    torch::Tensor stft_x = torch::abs(torch::stft(x, /*n_fft=*/n_fft,
        /*hop_length=*/hop_length,
        /*win_length=*/win_length,
        /*window=*/window,
        /*center=*/center,
        /*pad_mode=*/"constant",
        /*normalized=*/false,
        /*onesided=*/true,
        /*return_complex*/true));

    // Compute energy by summing over the last dimension and performing cumulative sum
    torch::Tensor energy = torch::cumsum(torch::sum(stft_x, /*dim=*/-1), /*dim=*/0);

    // Find the cutoff index
    return _find_cutoff_np(energy, 0.985);
}

// Main post-processing function
static torch::Tensor postprocessing(torch::Tensor out_batch, torch::Tensor x_batch, int64_t n_fft = 2048, int64_t hop_length = -1, int64_t win_length = -1, bool center = true) {
    
    // Default `win_length` to `n_fft` if not provided
    if (win_length == -1) {
        win_length = n_fft;
    }

    // Default `hop_length` to `win_length // 4` if not provided (like librosa)
    if (hop_length == -1) {
        hop_length = win_length / 4;
    }

    for (int64_t i = 0; i < out_batch.size(0); ++i) {
        torch::Tensor out = out_batch[i][0];
        torch::Tensor x = x_batch[i][0].cpu();

        //std::cout << "calling _get_cutoff_index_np with x of shape = " << x.sizes() << std::endl;

        // Get cutoff index using the helper function
        int64_t cutoffratio = _get_cutoff_index_np(x, n_fft, hop_length, win_length, center);

        int64_t length = out.size(0);

        // Create a Hann window (default in librosa)
        torch::Tensor window = torch::hann_window(win_length == -1 ? n_fft : win_length, torch::kFloat32);

        // Compute STFTs
        torch::Tensor stft_gt = torch::stft(x, n_fft, hop_length, win_length, window, center, "constant", false, true, true);
        torch::Tensor stft_out = torch::stft(out, n_fft, hop_length, win_length, window, center, "constant", false, true, true);

        // Compute energy ratio
        double energy_gt = torch::sum(torch::abs(stft_gt.index({ cutoffratio, torch::indexing::Slice() }))).item<double>();
        double energy_out = torch::sum(torch::abs(stft_out.index({ cutoffratio, torch::indexing::Slice() }))).item<double>();
        double energy_ratio = energy_gt / energy_out;

        // Clamp energy_ratio between 0.8 and 1.2
        energy_ratio = std::min(std::max(energy_ratio, 0.8), 1.2);

        // Adjust STFT output based on energy ratio and cutoff
        stft_out.index({ torch::indexing::Slice(torch::indexing::None, cutoffratio), torch::indexing::Slice() }) =
            stft_gt.index({ torch::indexing::Slice(torch::indexing::None, cutoffratio), torch::indexing::Slice() }) / energy_ratio;

        // Perform inverse STFT (ISTFT)
        torch::Tensor out_renewed = torch::istft(stft_out, n_fft, hop_length, win_length, window, center, false, ::std::nullopt, length);

        //std::cout << "out_renewed size = " << out_renewed.sizes() << std::endl;
        //std::cout << "out_batch size = " << out_batch.sizes() << std::endl;

        // Update the output batch
        out_batch[i] = out_renewed;
    }

    return out_batch;
}

struct DDPMLatentDiffusion::DDPMLatentDiffusionIntermediate
{
   //common across stages
   torch::Generator gen;

   //stage 1 output
   torch::Tensor input_z;
   torch::Tensor input_c;

   //stage 2 output
   torch::Tensor samples;
   
};

std::shared_ptr< DDPMLatentDiffusion::DDPMLatentDiffusionIntermediate > DDPMLatentDiffusion::generate_batch_stage1(Batch& batch, int64_t seed)
{
   auto intermediate = std::make_shared< DDPMLatentDiffusionIntermediate >();

   intermediate->gen = at::detail::createCPUGenerator();
   intermediate->gen.set_current_seed(seed);

   auto input = _get_input(batch, intermediate->gen);

   intermediate->input_z = input.first;
   intermediate->input_c = input.second;

   return intermediate;
}

std::shared_ptr< DDPMLatentDiffusion::DDPMLatentDiffusionIntermediate > DDPMLatentDiffusion::generate_batch_stage2(std::shared_ptr< DDPMLatentDiffusion::DDPMLatentDiffusionIntermediate > intermediate,
   int64_t ddim_steps, double unconditional_guidance_scale,
   std::optional< CallbackParams > callback_params)
{

   if (!intermediate)
   {
      throw std::runtime_error("generate_batch_stage2: intermediate is null.");
   }

   if (!intermediate->input_c.defined())
   {
      throw std::runtime_error("generate_batch_stage2: C is not defined. make sure stage 1 was run.");
   }

   if (!intermediate->input_z.defined())
   {
      throw std::runtime_error("generate_batch_stage2: Z is not defined. make sure stage 1 was run.");
   }

   auto z = intermediate->input_z;
   auto c = intermediate->input_c;

   std::optional< torch::Tensor > unconditional_cond;
   if (unconditional_guidance_scale != 1.0)
   {
      // In the python version, this is created inside VAEFeatureExtract, using:
      // self.unconditional_cond = -11.4981 + vae_embed[0].clone() * 0.0.
      unconditional_cond = torch::full(c.sizes(), -11.4981);
   }

   intermediate->samples = _sampler->sample(ddim_steps, c, unconditional_guidance_scale, intermediate->gen, unconditional_cond, callback_params);

   return intermediate;
}

torch::Tensor DDPMLatentDiffusion::generate_batch_stage3(std::shared_ptr< DDPMLatentDiffusion::DDPMLatentDiffusionIntermediate > intermediate,
   Batch& batch)
{
   if (!intermediate)
   {
      throw std::runtime_error("generate_batch_stage3: intermediate is null.");
   }

   if (!intermediate->samples.defined())
   {
      throw std::runtime_error("generate_batch_stage3: samples is not defined. make sure stage 2 was run.");
   }

   auto samples = intermediate->samples;

   auto mel = _decode_first_stage(samples);

   mel = _mel_replace_ops(mel, batch.lowpass_mel);

   //dump_tensor(mel, "mel_ov.raw");

   auto waveform = _mel_spectrogram_to_waveform(mel);

   //dump_tensor(waveform, "waveform_ov.raw");

   waveform = postprocessing(waveform, batch.waveform_lowpass);

   //dump_tensor(waveform, "waveform_pp_ov.raw");

   //auto max_amp = torch::max(torch::abs(waveform), -1);

   // Get the maximum amplitude along the last axis
   auto max_amp_ret = torch::max(torch::abs(waveform), /*dim=*/-1, /*keepdim=*/true);
   torch::Tensor max_amp = std::get<0>(max_amp_ret);

   // Normalize waveform by maximum amplitude, scale by 0.5
   waveform = 0.5 * waveform / max_amp;

   // Compute the mean amplitude along the last axis and keep dimensions
   torch::Tensor mean_amp = torch::mean(waveform, /*dim=*/-1, /*keepdim=*/true);

   // Subtract mean amplitude from waveform
   waveform = waveform - mean_amp;

   //dump_tensor(waveform, "waveform_ret_ov.raw");

   return waveform;
}

torch::Tensor DDPMLatentDiffusion::generate_batch(Batch& batch, int64_t seed,
   int64_t ddim_steps, double unconditional_guidance_scale,
   std::optional< CallbackParams > callback_params)
{
   auto intermediate = generate_batch_stage1(batch, seed);

   intermediate = generate_batch_stage2(intermediate, ddim_steps, unconditional_guidance_scale, callback_params);

   return generate_batch_stage3(intermediate, batch);
}


torch::Tensor DDPMLatentDiffusion::_get_first_stage_encoding(torch::Tensor x, torch::Generator &gen)
{
    auto encoder_posterior = _first_stage_encoder->encode(x);
    auto z = encoder_posterior->sample(gen);

    return _scale_factor * z;
}

torch::Tensor DDPMLatentDiffusion::_decode_first_stage(torch::Tensor z)
{
    z = 1.f / _scale_factor * z;
    auto decoding = _first_stage_encoder->decode(z);

    return decoding;
}

torch::Tensor DDPMLatentDiffusion::apply_model(torch::Tensor x_noisy, int64_t t, torch::Tensor cond)
{
    //std::cout << "apply_model called!" << std::endl;

    //std::cout << "x_noisy_shape = " << x_noisy.sizes() << std::endl;
    //std::cout << "cond = " << cond.sizes() << std::endl;

    x_noisy = x_noisy.contiguous();
    cond = cond.contiguous();

    //dump_tensor(x_noisy, "x_noisy_ov.raw");
    //dump_tensor(cond, "cond_ov.raw");

    auto x_noisy_ov = wrap_torch_tensor_as_ov(x_noisy);
    auto cond_ov = wrap_torch_tensor_as_ov(cond);

    _ddpm_infer.set_tensor("x_noisy", x_noisy_ov);
    _ddpm_infer.set_tensor("cond", cond_ov);

    auto t_ov = _ddpm_infer.get_tensor("t");
    int64_t* pT = t_ov.data<int64_t>();
    *pT = t;

    _ddpm_infer.infer();

    auto x_recon_ov = _ddpm_infer.get_output_tensor();
    auto x_recon = wrap_ov_tensor_as_torch(x_recon_ov).clone();

    //dump_tensor(x_recon, "x_recon_ov.raw");

    return x_recon;
}

// Helper function to find the cutoff
static int64_t _find_cutoff(const torch::Tensor& x, double percentile = 0.95) {
    double target_percentile_value = x[-1].item<double>() * percentile;

    for (int64_t i = 1; i < x.size(0); ++i) {
        if (x[-i].item<double>() < target_percentile_value) {
            return x.size(0) - i;
        }
    }
    return 0;
}

// Main function to locate the cutoff frequency
static int64_t _locate_cutoff_freq(const torch::Tensor& stft, double percentile = 0.985) {
    // Calculate the magnitude of the STFT
    torch::Tensor magnitude = torch::abs(stft);

    // Compute the energy (cumulative sum over the sum of magnitudes along dimension 0)
    torch::Tensor energy = torch::cumsum(torch::sum(magnitude, /*dim=*/0), /*dim=*/0);

    // Call the helper function to find the cutoff
    return _find_cutoff(energy, percentile);
}

torch::Tensor DDPMLatentDiffusion::_mel_replace_ops(torch::Tensor samples, torch::Tensor input)
{
    for (int64_t i = 0; i < samples.size(0); ++i) {
        // Compute the cutoff melbin using the locate cutoff function
        int64_t cutoff_melbin = _locate_cutoff_freq(torch::exp(input[i]));

        // Replace the values in 'samples' up to the cutoff_melbin
        samples[i].index({ "...", torch::indexing::Slice(torch::indexing::None, cutoff_melbin) }) =
            input[i].index({ "...", torch::indexing::Slice(torch::indexing::None, cutoff_melbin)});
    }

    return samples;
}

torch::Tensor DDPMLatentDiffusion::_mel_spectrogram_to_waveform(torch::Tensor mel)
{
    if (mel.sizes().size() == 4)
    {
        mel = mel.squeeze(1);
    }

    mel = mel.permute({ 0, 2, 1 }).contiguous();

    //dump_tensor(mel, "mel_input_ov.raw");

    auto mel_ov = wrap_torch_tensor_as_ov(mel);

    _vocoder_infer.set_input_tensor(mel_ov);

    _vocoder_infer.infer();

    auto waveform_ov = _vocoder_infer.get_output_tensor();

    return wrap_ov_tensor_as_torch(waveform_ov).clone();
}


