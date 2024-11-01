#pragma once

#include "audiosr_common.h"

class AutoEncoder;
class DDIMSampler;
class DDPMLatentDiffusion
{
public:

    DDPMLatentDiffusion(std::shared_ptr< AudioSR_Config > config);

    // this runs the full pipeline 
    torch::Tensor generate_batch(Batch& batch, int64_t seed = 42, int64_t ddim_steps = 50,
       double unconditional_guidance_scale = 3.5,
       std::optional< CallbackParams > callback_params = {});

    
    torch::Tensor apply_model(torch::Tensor x_noisy, int64_t t, torch::Tensor cond);

    void set_config(AudioSR_Config config);

    struct DDPMLatentDiffusionIntermediate;
    std::shared_ptr< DDPMLatentDiffusionIntermediate > generate_batch_stage1(Batch& batch, int64_t seed = 42);

    std::shared_ptr< DDPMLatentDiffusionIntermediate > generate_batch_stage2(std::shared_ptr< DDPMLatentDiffusion::DDPMLatentDiffusionIntermediate > intermediate,
       int64_t ddim_steps = 50, double unconditional_guidance_scale = 3.5,
       std::optional< CallbackParams > callback_params = {});

    torch::Tensor generate_batch_stage3(std::shared_ptr< DDPMLatentDiffusion::DDPMLatentDiffusionIntermediate > intermediate,
       Batch& batch);


private:

    torch::Tensor _get_first_stage_encoding(torch::Tensor x, torch::Generator &gen);
    torch::Tensor _decode_first_stage(torch::Tensor z);
    torch::Tensor _mel_replace_ops(torch::Tensor samples, torch::Tensor input);
    torch::Tensor _mel_spectrogram_to_waveform(torch::Tensor mel);

    std::pair<torch::Tensor, torch::Tensor> _get_input(Batch& batch, torch::Generator &gen);
  
    std::shared_ptr< AudioSR_Config > _config;

    std::shared_ptr< AutoEncoder > _first_stage_encoder;

    ov::InferRequest _vae_feature_extract_infer;
    ov::InferRequest _ddpm_infer;
    ov::InferRequest _vocoder_infer;

    std::shared_ptr< DDIMSampler > _sampler;

    //dumped from LattentDiffusion self.scale factor
    // TODO: Double check that this is consistent between 'speech' and 'basic' models.
    float _scale_factor = 0.334240138530731201f;

    void _init_vae();
    void _init_ddpm();
    void _init_vocoder();
};
