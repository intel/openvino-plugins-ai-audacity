#include "ddim_sampler.h"
#include "ddpm_latent_diffusion.h"

DDIMSampler::DDIMSampler(DDPMLatentDiffusion* model)
    : _model(model)
{
    _register_schedule();

    _ddpm_num_timesteps = _num_timesteps;
}

static torch::Tensor make_beta_schedule(
    const std::string& schedule,
    int64_t n_timestep,
    double linear_start = 1e-4,
    double linear_end = 2e-2,
    double cosine_s = 8e-3
) {
    if (schedule == "linear") {
        auto betas = torch::linspace(
            std::sqrt(linear_start), std::sqrt(linear_end), n_timestep, torch::kFloat64
        ).pow(2);
        return betas;
    }
    else if (schedule == "cosine") {
        auto timesteps = (torch::arange(n_timestep + 1, torch::kFloat64) / n_timestep) + cosine_s;
        auto alphas = timesteps / (1 + cosine_s) * M_PI / 2;
        alphas = torch::cos(alphas).pow(2);
        alphas = alphas / alphas[0];
        auto betas = 1 - alphas.slice(0, 1, alphas.size(0)) / alphas.slice(0, 0, alphas.size(0) - 1);
        return betas;
    }
    else if (schedule == "sqrt_linear") {
        auto betas = torch::linspace(linear_start, linear_end, n_timestep, torch::kFloat64);
        return betas;
    }
    else if (schedule == "sqrt") {
        auto betas = torch::linspace(linear_start, linear_end, n_timestep, torch::kFloat64).sqrt();
        return betas;
    }
    else {
        throw std::invalid_argument("schedule '" + schedule + "' unknown.");
    }
}

// In the python version, this is found within DDPM class, but moving it here since it seems
// like the stuff created here is mainly used by DDIMSampler. But perhaps we'll move it to 
// DDPM if we support more samplers than DDIM.
void DDIMSampler::_register_schedule()
{
    //DDPM: register_schedule called.
    //beta_schedule =  cosine
    //timesteps = 1000
    //linear_start = 0.0015
    //linear_end = 0.0195
    //cosine_s = 0.008
    //exists(given_betas) = False

    std::string beta_schedule = "cosine";
    int64_t timesteps = 1000;
    _num_timesteps = timesteps;
    double linear_start = 0.0015;
    double linear_end = 0.0195;
    double cosine_s = 0.008;

    auto betas = make_beta_schedule(beta_schedule, timesteps, linear_start, linear_end, cosine_s);
    auto alphas = 1.0 - betas;
    auto alphas_cumprod = torch::cumprod(alphas, 0);
    auto alphas_cumprod_prev = torch::cat({ torch::tensor({1.0}, alphas.options()), alphas_cumprod.slice(0, 0, -1) });

    std::cout << "betas shape = " << betas.sizes() << std::endl;

    _linear_start = linear_start;
    _linear_end = linear_end;

    std::cout << "alphas_cumprod shape = " << alphas_cumprod.sizes() << std::endl;

    _betas = betas.to(torch::kFloat32);
    _alphas_cumprod = alphas_cumprod.to(torch::kFloat32);
    _alphas_cumprod_prev = alphas_cumprod_prev.to(torch::kFloat32);

    _sqrt_alphas_cumprod = torch::sqrt(alphas_cumprod).to(torch::kFloat32);
    _sqrt_one_minus_alphas_cumprod = torch::sqrt(1.0 - alphas_cumprod).to(torch::kFloat32);
    _log_one_minus_alphas_cumprod = torch::log(1.0 - alphas_cumprod).to(torch::kFloat32);;
    _sqrt_recip_alphas_cumprod = torch::sqrt(1.0 / (alphas_cumprod)).to(torch::kFloat32);
    _sqrt_recipm1_alphas_cumprod = torch::sqrt(1.0 / (alphas_cumprod) - 1).to(torch::kFloat32);

    
    double v_posterior = 0.0;
    // calculations for posterior q(x_{t-1} | x_t, x_0)
    auto posterior_variance = (1 - v_posterior) * betas * (
        1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + v_posterior * betas;
    //above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
    _posterior_variance = posterior_variance.to(torch::kFloat32);

    _posterior_log_variance_clipped = torch::log(torch::clamp(posterior_variance, 1e-20)).to(torch::kFloat32);

    _posterior_mean_coef1 = (betas * torch::sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).to(torch::kFloat32);

    _posterior_mean_coef2 = ((1.0 - alphas_cumprod_prev) * torch::sqrt(alphas) / (1.0 - alphas_cumprod)).to(torch::kFloat32);

    //elif self.parameterization == "v":
    // side note: I have no idea why such a complicated formula is wrapped into a call to 'ones_like'.. as 
    auto lvlb_weights = torch::ones_like(
        (_betas.pow(2)) / (2 * _posterior_variance * alphas.to(torch::kFloat32) * (1 - _alphas_cumprod))
    );

    std::cout << "lvlb_weights shape = " << lvlb_weights.sizes() << std::endl;
}

static torch::Tensor make_ddim_timesteps(
    const std::string& ddim_discr_method,
    int num_ddim_timesteps,
    int num_ddpm_timesteps
) {
    torch::Tensor ddim_timesteps;

    if (ddim_discr_method == "uniform") {
        // c = num_ddpm_timesteps // num_ddim_timesteps
        int c = num_ddpm_timesteps / num_ddim_timesteps;

        // ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
        std::vector<int64_t> timesteps_vec;
        for (int i = 0; i < num_ddpm_timesteps; i += c) {
            timesteps_vec.push_back(i);
        }
        ddim_timesteps = torch::tensor(timesteps_vec, torch::kInt64);

    }
    else if (ddim_discr_method == "quad") {
        // ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2).astype(int)
        auto linspace = torch::linspace(0, std::sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps, torch::kFloat64);
        ddim_timesteps = linspace.pow(2).to(torch::kInt64);

    }
    else {
        throw std::runtime_error("There is no ddim discretization method called \"" + ddim_discr_method + "\"");
    }

    // add one to get the final alpha values right
    auto steps_out = ddim_timesteps + 1;

    return steps_out;
}


static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> make_ddim_sampling_parameters(
    const torch::Tensor& alphacums,
    torch::Tensor ddim_timesteps,
    double eta,
    bool verbose = false
) {

    // select alphas for computing the variance schedule
    auto alphas = torch::take(alphacums, ddim_timesteps);

    //std::cout << "ddim_timesteps = " << std::endl;
    //std::cout << ddim_timesteps << std::endl;

    //std::cout << "alphas = " << std::endl;
    //std::cout << alphas << std::endl;


    // alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    std::vector<double> alphas_prev_vec = { alphacums[0].item<float>() };
    auto timesteps_sliced = ddim_timesteps.slice(0, 0, -1);
    auto alphas_prev_sliced = torch::take(alphacums, timesteps_sliced);
    for (int64_t i = 0; i < alphas_prev_sliced.size(0); ++i) {
        alphas_prev_vec.push_back(alphas_prev_sliced[i].item<float>());
    }
    auto alphas_prev = torch::tensor(alphas_prev_vec, alphacums.options());

    // sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    auto sigmas = eta * torch::sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev));

    if (verbose) {
        std::cout << "Selected alphas for ddim sampler: a_t: " << alphas << "; a_(t-1): " << alphas_prev << std::endl;
        std::cout << "For the chosen value of eta, which is " << eta << ", this results in the following sigma_t schedule for ddim sampler: " << sigmas << std::endl;
    }

    return std::make_tuple(sigmas, alphas, alphas_prev);
}


void DDIMSampler::_make_schedule(int64_t ddim_num_steps, double ddim_eta)
{
    //self.ddim_timesteps = make_ddim_timesteps(
    //    ddim_discr_method = ddim_discretize,
    //    num_ddim_timesteps = ddim_num_steps,
    //    num_ddpm_timesteps = self.ddpm_num_timesteps,
    //    verbose = verbose,
    //    )
    
    // ddim_discretize =  uniform

    _ddim_timesteps = make_ddim_timesteps("uniform", ddim_num_steps, _ddpm_num_timesteps);

    auto alphas_cumprod = _alphas_cumprod;

    std::cout << "alphas_cumprod shape = " << alphas_cumprod.sizes() << std::endl;

    auto [ddim_sigmas, ddim_alphas, ddim_alphas_prev] = make_ddim_sampling_parameters(alphas_cumprod, _ddim_timesteps, ddim_eta);

    _ddim_sigmas = ddim_sigmas;
    _ddim_alphas = ddim_alphas;
    _ddim_alphas_prev = ddim_alphas_prev;
    _ddim_sqrt_one_minus_alphas = torch::sqrt(1.f - _ddim_alphas);

    _ddim_sigmas_for_original_num_steps = ddim_eta * torch::sqrt(
        (1 - _alphas_cumprod_prev)
        / (1 - _alphas_cumprod)
        * (1 - _alphas_cumprod / _alphas_cumprod_prev)
    );
 
}

torch::Tensor DDIMSampler::sample(int64_t ddim_num_steps,
    torch::Tensor conditioning,
    double unconditional_guidance_scale,
   torch::Generator &gen,
    std::optional< torch::Tensor > unconditional_conditioning,
   std::optional< CallbackParams > callback_params)
{
    _make_schedule(ddim_num_steps, 1.0);

    std::vector<int64_t> shape = { 1, 16, 128, 32 };

    auto C = shape[1];
    auto H = shape[2];
    auto W = shape[3];

    //ddim_sampling starts here.
    size_t b = 1;

    auto img = torch::randn(shape, gen);

    auto timesteps = _ddim_timesteps;

    auto total_steps = timesteps.size(0);

    for (size_t i = 0; i < total_steps; i++)
    {
        auto index = total_steps - i - 1;
        auto step = timesteps[index];

        auto sample_ddim_ret = _p_sample_ddim(img, conditioning, step.item<int64_t>(), index,
           unconditional_guidance_scale, gen, unconditional_conditioning, callback_params);

        img = sample_ddim_ret.first;
        auto pred_x0 = sample_ddim_ret.second;

        if (callback_params && callback_params->callback )
        {
           callback_params->callback(i, callback_params->user);
        }
    }

    return img;
}


static torch::Tensor extract_into_tensor(const torch::Tensor& a, const torch::Tensor& t, const torch::IntArrayRef x_shape) {
    // Get the first dimension of tensor `t`
    int64_t b = t.size(0);

    // Gather values from `a` at indices specified by `t` along the last dimension (-1)
    torch::Tensor out = a.gather(-1, t).contiguous();

    // Reshape the output tensor
    std::vector<int64_t> new_shape = { b };
    new_shape.insert(new_shape.end(), x_shape.size() - 1, 1);

    // Return reshaped tensor
    return out.view(new_shape).contiguous();
}


torch::Tensor DDIMSampler::_predict_eps_from_z_and_v(torch::Tensor x_t, torch::Tensor t, torch::Tensor v)
{
    return (
        extract_into_tensor(_sqrt_alphas_cumprod, t, x_t.sizes()) * v 
        + extract_into_tensor(_sqrt_one_minus_alphas_cumprod, t, x_t.sizes())
        * x_t
        );
}

torch::Tensor DDIMSampler::_predict_start_from_z_and_v(torch::Tensor x_t, torch::Tensor t, torch::Tensor v)
{
    return (
        extract_into_tensor(_sqrt_alphas_cumprod, t, x_t.sizes()) * x_t
        - extract_into_tensor(_sqrt_one_minus_alphas_cumprod, t, x_t.sizes()) * v
        );
}

std::pair<torch::Tensor, torch::Tensor> DDIMSampler::_p_sample_ddim(torch::Tensor x, torch::Tensor c, int64_t t, int64_t index,
    double unconditional_guidance_scale,
    torch::Generator &gen,
    std::optional< torch::Tensor > unconditional_conditioning,
    std::optional< CallbackParams > callback_params)
{
    float temperature = 1.f;

    torch::Tensor model_output;
    if (!unconditional_conditioning || unconditional_guidance_scale == 1.0)
    {
        throw std::runtime_error("(!unconditional_conditioning || unconditional_guidance_scale == 1.0) case not implmented yet!");
    }
    else
    {
        auto x_in = x;
        auto t_in = t;

        auto model_t = _model->apply_model(x_in, t_in, c);
        
        auto model_uncond = _model->apply_model(x_in, t_in, *unconditional_conditioning);

        model_output = model_uncond + unconditional_guidance_scale * (
            model_t - model_uncond
            );
    }

    //dump_tensor(model_output, "model_output_ov.raw");

    auto t_tensor = torch::tensor({ t }, torch::dtype(torch::kInt64));

    auto e_t = _predict_eps_from_z_and_v(x, t_tensor, model_output);

    //dump_tensor(e_t, "e_t_ov.raw");

    auto& alphas = _ddim_alphas;
    auto& alphas_prev = _ddim_alphas_prev;
    auto& sqrt_one_minus_alphas = _ddim_sqrt_one_minus_alphas;
    auto& sigmas = _ddim_sigmas;

    // select parameters corresponding to the currently considered timestep
    int64_t b = 1;
    auto a_t = alphas[index];
    auto a_prev = alphas_prev[index];
    auto sigma_t = sigmas[index];
    auto sqrt_one_minus_at = sqrt_one_minus_alphas[index];

    //current prediction for x_0
    auto pred_x0 = _predict_start_from_z_and_v(x, t_tensor, model_output);

    //direction pointing to x_t
    auto dir_xt = (1.0 - a_prev - sigma_t.pow(2)).sqrt() * e_t;

    //TODO: use generator for randn
    auto noise = sigma_t * torch::randn(x.sizes(), gen) * temperature;

    auto x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise;

    //dump_tensor(x_prev, "x_prev_ov.raw");
    //dump_tensor(pred_x0, "pred_x0_ov.raw");

    return { x_prev, pred_x0 };
}
