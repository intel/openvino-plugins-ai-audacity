// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include <demix/mdx23c.h>
#include "musicgen/musicgen_utils.h"

namespace ov_demix
{

   struct STFTConfig {
      int n_fft;
      int hop_length;
      int dim_f;
   };

   class MDX23C::STFT {
   public:
      STFT(const STFTConfig& config)
         : n_fft(config.n_fft),
         hop_length(config.hop_length),
         dim_f(config.dim_f),
         window(torch::hann_window(config.n_fft, /*periodic=*/true, torch::TensorOptions().dtype(torch::kFloat))) {
      }

      torch::Tensor apply(const torch::Tensor& input) {
         auto x = input;
         auto window_on_device = window.to(x.device());

         auto batch_dims = x.sizes().slice(0, x.dim() - 2);
         int64_t c = x.size(-2);
         int64_t t = x.size(-1);

         x = x.view({ -1, t });
         auto stft_result = torch::stft(x, n_fft,
            hop_length,
            /*win_length*/n_fft,
            /*window*/window_on_device,
            /*center*/true,
            /*pad_mode*/"reflect",
            /*normalized*/false,
            /*onesided*/true,
            /*return_complex*/true);

         auto x_real = torch::view_as_real(stft_result);  // shape: [B, F, T, 2]
         x_real = x_real.permute({ 0, 3, 1, 2 });         // -> [B, 2, F, T]

         auto total_channels = c * 2;
         int64_t f = x_real.size(2);
         int64_t t_new = x_real.size(3);
         auto shape = batch_dims.vec();
         shape.push_back(total_channels);
         shape.push_back(f);
         shape.push_back(t_new);
         x_real = x_real.reshape(shape);
         return x_real.index({ "...", torch::indexing::Slice(0, dim_f), torch::indexing::Slice() });
      }

      torch::Tensor inverse(torch::Tensor x) {
         auto window_on_device = window.to(x.device());
         auto batch_dims = x.sizes().slice(0, x.dim() - 3);
         int64_t c = x.size(-3);
         int64_t f = x.size(-2);
         int64_t t = x.size(-1);
         int64_t n = n_fft / 2 + 1;

         auto shape_pad = batch_dims.vec();
         shape_pad.push_back(c);
         shape_pad.push_back(n - f);
         shape_pad.push_back(t);

         auto f_pad = torch::zeros(shape_pad, x.options());
         x = torch::cat({ x, f_pad }, -2);  // pad frequency dim

         x = x.reshape({ -1, 2, n, t });
         x = x.permute({ 0, 2, 3, 1 });  // -> [B, F, T, 2]
         auto x_complex = torch::complex(x.index({ "...", 0 }), x.index({ "...", 1 }));

         auto result = torch::istft(x_complex,
            /*n_fft*/n_fft,
            /*hop_length*/hop_length,
            /*win_length*/n_fft,
            /*window*/window_on_device,
            /*center*/true);

         auto shape_final = batch_dims.vec();
         shape_final.push_back(2);
         shape_final.push_back(result.size(-1));
         return result.view(shape_final);
      }

   private:
      int n_fft;
      int hop_length;
      int dim_f;
      torch::Tensor window;
   };

   MDX23C::MDX23C(const std::string& model_dir,
        const std::string& device,
        const std::string& cache_dir)
    {
        ov::Core core;

        {
            auto xml_path = FullPath(model_dir, "mdx23c_fwd.xml");
            auto model = core.read_model(xml_path);
            std::cout << "FWD:" << std::endl;
            logBasicModelInfo(model);

            ov::AnyMap properties = { ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY) };

#if 1
            // Enable f32 precision for GPU device, as for some reason it produces incorrect results
            // for default fp16 mode right now.
            if (device.find("GPU") != std::string::npos)
            {
                properties.insert(ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
            }
#endif

            if (!cache_dir.empty())
            {
                properties.insert(ov::cache_dir(cache_dir));
            }

            auto compiled_model = core.compile_model(model, device, properties);
            _forward_ir = compiled_model.create_infer_request();

            auto out_shape = _forward_ir.get_output_tensor().get_shape();
            if (out_shape.size() != 5)
            {
               throw std::runtime_error("MDX23C: Expected output tensor to have rank 5, but it is "
                  + std::to_string(out_shape.size()));
            }

            _num_instruments = out_shape[1];

            std::cout << "_num_instruments = " << _num_instruments << std::endl;

            STFTConfig stft_config;
            stft_config.n_fft = 2048;
            stft_config.hop_length = 512;
            stft_config.dim_f = 1024;

            _stft = std::make_shared<STFT>(stft_config);
        }
    }

    torch::Tensor MDX23C::run(torch::Tensor arr)
    {
        auto x = _stft->apply(arr);

        _forward_ir.set_input_tensor(wrap_torch_tensor_as_ov(x.contiguous()));

        using namespace std::chrono;
        using Clock = std::chrono::high_resolution_clock;
        uint64_t t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        _forward_ir.infer();
        uint64_t t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        std::cout << "inference time = " << t1 - t0 << std::endl;

        auto fwd_out = wrap_ov_tensor_as_torch(_forward_ir.get_output_tensor());

        auto ret = _stft->inverse(fwd_out);

        return ret;
    }
}
