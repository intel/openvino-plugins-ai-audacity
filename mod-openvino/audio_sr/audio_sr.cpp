
#include "audio_sr.h"
#include "ddpm_latent_diffusion.h"

AudioSR::AudioSR(std::string model_folder,
   std::string first_stage_encoder_device,
   std::string vae_feature_extract_device,
   std::string ddpm__device,
   std::string vocoder_device,
   AudioSRModel model_selection,
   std::string cache_dir)
{
   //The python version calculates this as:
   // from librosa.filters import mel as librosa_mel_fn
   // mel = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=n_mel, fmin=mel_fmin, fmax=mel_fmax)
   // But as this seems to just be a constant, we'll just pre-compute it and read it from disk.
   _mel_basis = read_tensor(FullPath(model_folder, "mel_24000_cpu.raw"), { 256, 1025 });


   _audio_sr_config = std::make_shared< AudioSR_Config >();
   _audio_sr_config->model_folder = model_folder;
   _audio_sr_config->first_stage_encoder_device = first_stage_encoder_device;
   _audio_sr_config->vae_feature_extract_device = vae_feature_extract_device;
   _audio_sr_config->ddpm__device = ddpm__device;
   _audio_sr_config->vocoder_device = vocoder_device;
   _audio_sr_config->model_selection = model_selection;

   if (!cache_dir.empty())
   {
      _audio_sr_config->core.set_property(ov::cache_dir(cache_dir));
   }

   _ddpm_latent_diffusion = std::make_shared< DDPMLatentDiffusion >(_audio_sr_config);
}

void AudioSR::set_config(AudioSR_Config config)
{
   _ddpm_latent_diffusion->set_config(config);
}

static inline torch::Tensor pad_wav(torch::Tensor waveform, int64_t target_length) {
   int64_t waveform_length = waveform.size(-1);

   if (waveform_length == target_length) {
      return waveform;
   }

   torch::Tensor temp_wav = torch::zeros({ 1, target_length }, torch::kFloat32);
   temp_wav.index({ 0, torch::indexing::Slice(0, waveform_length) }) = waveform.squeeze(0);

   return temp_wav;
}

static inline torch::Tensor dynamic_range_compression_torch(torch::Tensor x, float C = 1.0, float clip_val = 1e-5)
{
   torch::Tensor clamped_x = torch::clamp(x, clip_val, std::numeric_limits<float>::max());
   return torch::log(clamped_x * C);
}

static inline torch::Tensor spectral_normalize_torch(torch::Tensor magnitudes)
{
   auto output = dynamic_range_compression_torch(magnitudes);
   return output;
}

std::pair< torch::Tensor, torch::Tensor> AudioSR::_mel_spectrogram_train(torch::Tensor y)
{
   int64_t sampling_rate = 48000;
   int64_t filter_length = 2048;
   int64_t hop_length = 480;
   int64_t win_length = 2048;
   int64_t n_mel = 256;
   int64_t mel_fmin = 20;
   int64_t mel_fmax = 24000;

   
   auto mel_basis = _mel_basis;

   auto hann_window = torch::hann_window(win_length);

   y = torch::nn::functional::pad(
      y.unsqueeze(1), // Unsqueeze along dim 1
      torch::nn::functional::PadFuncOptions({ (filter_length - hop_length) / 2, (filter_length - hop_length) / 2 }).mode(torch::kReflect)
   ).squeeze(1);

   auto stft_spec = torch::stft(y, filter_length, hop_length, win_length, hann_window, false, "reflect", false, true, true);

   stft_spec = torch::abs(stft_spec);

   auto mel = spectral_normalize_torch(torch::matmul(mel_basis, stft_spec));

   return { mel.index({0}), stft_spec.index({0}) };
}

torch::Tensor pad_spec(torch::Tensor log_mel_spec, int64_t target_frame) {
   // Get the number of frames in the input tensor (assuming it's 2D: [n_frames, n_features])
   int64_t n_frames = log_mel_spec.size(0);
   int64_t p = target_frame - n_frames;

   // Cut and pad
   if (p > 0) {
      // ZeroPad2d equivalent: (0, 0, 0, p) means pad last dimension with 0 and frame dimension with 'p'
      log_mel_spec = torch::constant_pad_nd(log_mel_spec, { 0, 0, 0, p }, 0);  // Pad with 0
   }
   else if (p < 0) {
      // Slice the tensor to retain only the first `target_frame` frames
      log_mel_spec = log_mel_spec.slice(0, 0, target_frame);  // Slice along the first dimension
   }

   // Check if the last dimension's size is odd and remove the last column if true
   if (log_mel_spec.size(-1) % 2 != 0) {
      log_mel_spec = log_mel_spec.slice(-1, 0, log_mel_spec.size(-1) - 1);  // Slice off the last column
   }

   return log_mel_spec;
}

std::pair<torch::Tensor, torch::Tensor> AudioSR::_wav_feature_extraction(torch::Tensor waveform, int64_t target_frame)
{
   auto mel_spectrogram_train_ret = _mel_spectrogram_train(waveform);

   auto log_mel_spec = mel_spectrogram_train_ret.first;
   auto stft = mel_spectrogram_train_ret.second;

   log_mel_spec = log_mel_spec.t();
   stft = stft.t();

   log_mel_spec = pad_spec(log_mel_spec, target_frame);
   stft = pad_spec(stft, target_frame);

   return { log_mel_spec, stft };
}

// Helper function equivalent to _find_cutoff
int64_t find_cutoff(const torch::Tensor& x, double percentile = 0.95) {
   // Calculate the percentile value
   auto percentile_value = x[-1].item<double>() * percentile;

   // Find the cutoff point
   for (int64_t i = 1; i < x.size(0); i++) {
      if (x[-i].item<double>() < percentile_value) {
         std::cout << "find_cutoff returning " << x.size(0) - i << std::endl;
         return x.size(0) - i;
      }
   }

   return 0;
}

// Main function equivalent to _locate_cutoff_freq
int64_t locate_cutoff_freq(const torch::Tensor& stft, double percentile = 0.97) {
   // Compute the magnitude of the STFT
   torch::Tensor magnitude = torch::abs(stft);

   // Compute the energy along the frequency dimension (dim=0 in this case)
   torch::Tensor energy = torch::cumsum(torch::sum(magnitude, /*dim=*/0), /*dim=*/0);

   // Call the helper function to find the cutoff
   return find_cutoff(energy, percentile);
}

void AudioSR::normalize_and_pad(float* pSamples, size_t num_samples, Batch &batch)
{
   if (!pSamples)
   {
      throw std::runtime_error("pSamples is NULL");
   }

   if (num_samples > nchunk_samples())
   {
      throw std::runtime_error("num_samples must be <= nchunk_samples");
   }

   //normalize it.
   auto waveform = torch::from_blob((void*)pSamples, { static_cast<long>(num_samples) }, torch::kFloat);
   waveform = waveform - waveform.mean();
   waveform = waveform / (waveform.abs().max().item<float>() + 1e-8);
   waveform = waveform * 0.5f;

   auto input_wav_tensor = waveform.unsqueeze(0);

   int64_t n_padded_samples = nchunk_samples();

   if (input_wav_tensor.size(-1) != n_padded_samples)
   {
      int64_t padded_length = ((input_wav_tensor.size(-1) / n_padded_samples) + 1) * n_padded_samples;

      input_wav_tensor = pad_wav(input_wav_tensor, padded_length);
   }

   double pad_duration = input_wav_tensor.size(-1) / 48000.0;
   int64_t target_frame = (int64_t)(pad_duration * 100);

   input_wav_tensor = input_wav_tensor.contiguous();

   auto wav_feature_extraction_ret = _wav_feature_extraction(input_wav_tensor, target_frame);
   auto log_mel_spec = wav_feature_extraction_ret.first;
   auto stft = wav_feature_extraction_ret.second;

   batch.target_frame = target_frame;
   batch.waveform = input_wav_tensor.unsqueeze(0);
   batch.sampling_rate = 48000;
   batch.duration = pad_duration;
   batch.stft = stft;
   batch.log_mel_spec = log_mel_spec.unsqueeze(0);

   batch.cutoff_freq = (locate_cutoff_freq(batch.stft, 0.985) / 1024.0) * 24000.0;

   // If the audio is almost empty.Give up processing
   if (batch.cutoff_freq < 1000.0)
      batch.cutoff_freq = 24000.0;

   batch.stft = batch.stft.unsqueeze(0);
}


torch::Tensor AudioSR::run_audio_sr(Batch& batch,
   double unconditional_guidance_scale,
   int ddim_steps,
   int64_t seed,
   std::optional< CallbackParams > callback_params
)
{
   auto intermediate = run_audio_sr_stage1(batch, seed);

   intermediate = run_audio_sr_stage2(intermediate, unconditional_guidance_scale, ddim_steps, seed, callback_params);

   return run_audio_sr_stage3(intermediate, batch);
}

struct AudioSR::AudioSRIntermediate
{
   std::shared_ptr< DDPMLatentDiffusion::DDPMLatentDiffusionIntermediate > intermediate;
};

std::shared_ptr< AudioSR::AudioSRIntermediate > AudioSR::run_audio_sr_stage1(Batch& batch, int64_t seed)
{
   auto intermediate = std::make_shared< AudioSR::AudioSRIntermediate >();

   auto lowpass_feature_extraction_ret = _wav_feature_extraction(batch.waveform_lowpass, batch.target_frame);

   batch.lowpass_mel = lowpass_feature_extraction_ret.first.unsqueeze(0);
   batch.waveform_lowpass = batch.waveform_lowpass.unsqueeze(0);

   intermediate->intermediate = _ddpm_latent_diffusion->generate_batch_stage1(batch, seed);

   return intermediate;
}

std::shared_ptr< AudioSR::AudioSRIntermediate > AudioSR::run_audio_sr_stage2(std::shared_ptr< AudioSR::AudioSRIntermediate > intermediate,
   double unconditional_guidance_scale,
   int ddim_steps,
   int64_t seed,
   std::optional< CallbackParams > callback_params)
{
   if (!intermediate)
   {
      throw std::runtime_error("AudioSR::run_audio_sr_stage2: intermediate is null!");
   }

   if (!intermediate->intermediate)
   {
      throw std::runtime_error("AudioSR::run_audio_sr_stage2: latent diffusion intermediate is null!");
   }

   intermediate->intermediate = _ddpm_latent_diffusion->generate_batch_stage2(intermediate->intermediate,
      ddim_steps, unconditional_guidance_scale, callback_params);

   return intermediate; 
}

torch::Tensor AudioSR::run_audio_sr_stage3(std::shared_ptr< AudioSR::AudioSRIntermediate > intermediate, Batch& batch)
{
   if (!intermediate)
   {
      throw std::runtime_error("AudioSR::run_audio_sr_stage3: intermediate is null!");
   }

   if (!intermediate->intermediate)
   {
      throw std::runtime_error("AudioSR::run_audio_sr_stage3: latent diffusion intermediate is null!");
   }

   return _ddpm_latent_diffusion->generate_batch_stage3(intermediate->intermediate, batch);
}
