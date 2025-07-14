// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include <demix/apollo.h>
#include "musicgen/musicgen_utils.h"

namespace ov_demix
{
    Apollo::Apollo(const std::string& model_dir,
        const std::string& device,
        const std::string& cache_dir)
    {
        ov::Core core;

        {
            auto xml_path = FullPath(model_dir, "apollo_fwd.xml");
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
            if (out_shape.size() != 4)
            {
                throw std::runtime_error("Apollo: Expected output tensor to have rank 4, but it is "
                    + std::to_string(out_shape.size()));
            }

            _hop_len = out_shape[2] - 1;
            _win_len = _hop_len * 2;

            int64_t num_frames = static_cast<int64_t>(out_shape[3]);
            _chunk_len = (num_frames - 1) * _hop_len - 2 * (_win_len / 2) + _win_len;

            std::cout << "Apollo: _hop_len = " << _hop_len << std::endl;
            std::cout << "Apollo: _win_len = " << _win_len << std::endl;
            std::cout << "Apollo: _chunk_len = " << _chunk_len << std::endl;

            _window = torch::hann_window(_win_len, torch::TensorOptions().dtype(torch::kFloat32));
        }
    }

    torch::Tensor Apollo::run(torch::Tensor arr)
    {
        auto arr_sizes = arr.sizes();
        TORCH_CHECK(arr_sizes.size() == 3, "Expected input of shape [B, nch, nsample]");
        int64_t B = arr_sizes[0];
        int64_t nch = arr_sizes[1];
        int64_t nsample = arr_sizes[2];

        auto arr_ov = wrap_torch_tensor_as_ov(arr);

        int64_t win_len = _win_len;
        int64_t hop_len = _hop_len;

        torch::Tensor spec;
        {
            torch::Tensor input_reshaped = arr.view({ B * nch, nsample });

            spec = torch::stft(input_reshaped,
                /*n_fft=*/win_len,
                /*hop_length=*/hop_len,
                /*win_length=*/win_len,
                /*window=*/_window,
                /*center=*/true,
                /*pad_mode=*/"reflect",
                /*normalized=*/false,
                /*onesided=*/true,
                /*return_complex=*/true);

            torch::Tensor spec_real = torch::view_as_real(spec);  // shape [B*nch, F, T, 2]
            spec = spec_real.contiguous();
        }

        _forward_ir.set_input_tensor(wrap_torch_tensor_as_ov(spec));

        using namespace std::chrono;
        using Clock = std::chrono::high_resolution_clock;
        uint64_t t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        _forward_ir.infer();
        uint64_t t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        std::cout << "inference time = " << t1 - t0 << std::endl;

        auto est_spec = wrap_ov_tensor_as_torch(_forward_ir.get_output_tensor());

        // est_spec shape: [B*nch, 2, F, T]
        // Split real and imag
        torch::Tensor real = est_spec.select(1, 0);
        torch::Tensor imag = est_spec.select(1, 1);
        auto stft_repr = torch::complex(real, imag);  // shape [B*nch, F, T]

        torch::Tensor output = torch::istft(stft_repr, /*n_fft=*/win_len,
            /*hop_length=*/hop_len,
            /*win_length=*/win_len,
            /*window=*/_window,
            /*center=*/true,
            /*normalized=*/false,
            /*onesided=*/true,
            /*length=*/c10::nullopt);  // Use nsample if needed

        output = output.view({ B, nch, -1 });
        return output;
    }
}
