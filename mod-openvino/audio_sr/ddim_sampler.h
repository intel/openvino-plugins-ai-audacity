#pragma once

#include "audiosr_common.h"

class DDPMLatentDiffusion;

class DDIMSampler
{
public:

    DDIMSampler(DDPMLatentDiffusion* model);

    torch::Tensor sample(int64_t ddim_num_steps,
        torch::Tensor conditioning,
        double unconditional_guidance_scale,
        torch::Generator &gen,
        std::optional< torch::Tensor > unconditional_conditioning,
        std::optional< CallbackParams > callback_params = {});

private:

    DDPMLatentDiffusion* _model;

    void _register_schedule();
    void _make_schedule(int64_t ddim_num_steps, double ddim_eta = 0.0);

    std::pair<torch::Tensor, torch::Tensor> _p_sample_ddim(torch::Tensor x, torch::Tensor c, int64_t t, int64_t index,
        double unconditional_guidance_scale,
        torch::Generator &gen,
        std::optional< torch::Tensor > unconditional_conditioning,
        std::optional< CallbackParams > callback_params = {});

    torch::Tensor _predict_eps_from_z_and_v(torch::Tensor x_t, torch::Tensor t, torch::Tensor v);
    torch::Tensor _predict_start_from_z_and_v(torch::Tensor x_t, torch::Tensor t, torch::Tensor v);


    int64_t _num_timesteps;
    int64_t _ddpm_num_timesteps;
    double _linear_start;
    double _linear_end;
    torch::Tensor _betas;
    torch::Tensor _alphas_cumprod;
    torch::Tensor _alphas_cumprod_prev;
    torch::Tensor _sqrt_alphas_cumprod;
    torch::Tensor _sqrt_one_minus_alphas_cumprod;
    torch::Tensor _log_one_minus_alphas_cumprod;
    torch::Tensor _sqrt_recip_alphas_cumprod;
    torch::Tensor _sqrt_recipm1_alphas_cumprod;
    torch::Tensor _posterior_variance;
    torch::Tensor _posterior_log_variance_clipped;
    torch::Tensor _posterior_mean_coef1;
    torch::Tensor _posterior_mean_coef2;

    torch::Tensor _ddim_timesteps;

    torch::Tensor _ddim_sigmas;
    torch::Tensor _ddim_alphas;
    torch::Tensor _ddim_alphas_prev;
    torch::Tensor _ddim_sqrt_one_minus_alphas;
    torch::Tensor _ddim_sigmas_for_original_num_steps;
};
