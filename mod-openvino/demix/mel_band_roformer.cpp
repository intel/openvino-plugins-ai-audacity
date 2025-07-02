// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include <demix/mel_band_roformer.h>
#include "musicgen/musicgen_utils.h"

namespace ov_demix
{
    MelBandRoformer::MelBandRoformer(const std::string& model_dir,
        const std::string& device,
        const std::string& cache_dir,
        DemixModel::PadMode pad_mode)
        : _pad_mode(pad_mode)
    {
        ov::Core core;

        {
            auto xml_path = FullPath(model_dir, "mel_band_pre.xml");
            auto model = core.read_model(xml_path);
            std::cout << "PRE:" << std::endl;
            logBasicModelInfo(model);

            auto compiled_model = core.compile_model(model, "CPU");
            _pre_ir = compiled_model.create_infer_request();
        }

        {
            auto xml_path = FullPath(model_dir, "mel_band_fwd.xml");
            auto model = core.read_model(xml_path);
            std::cout << "FWD:" << std::endl;
            logBasicModelInfo(model);

            ov::AnyMap properties = { ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY) };

            // Enable f32 precision for GPU device, as for some reason it produces incorrect results
            // for default fp16 mode right now.
            if (device.find("GPU") != std::string::npos)
            {
                properties.insert(ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
            }

            if (!cache_dir.empty())
            {
                properties.insert(ov::cache_dir(cache_dir));
            }

            auto compiled_model = core.compile_model(model, device, properties);
            _forward_ir = compiled_model.create_infer_request();
        }

        {
            auto xml_path = FullPath(model_dir, "mel_band_post.xml");
            auto model = core.read_model(xml_path);
            std::cout << "POST:" << std::endl;
            logBasicModelInfo(model);

            auto compiled_model = core.compile_model(model, "CPU");
            _post_ir = compiled_model.create_infer_request();
        }
    }

    torch::Tensor MelBandRoformer::run(torch::Tensor arr)
    {
        auto arr_ov = wrap_torch_tensor_as_ov(arr);

        _pre_ir.set_input_tensor(arr_ov);
        _pre_ir.infer();

        auto stft_repr_ov = _pre_ir.get_tensor("stft_repr");
        auto stft_window_ov = _pre_ir.get_tensor("stft_window");

        _forward_ir.set_input_tensor(stft_repr_ov);
        _forward_ir.infer();

        auto masks_ov = _forward_ir.get_tensor("masks");

        _post_ir.set_tensor("stft_repr", stft_repr_ov);
        _post_ir.set_tensor("masks", masks_ov);
        _post_ir.set_tensor("stft_window", stft_window_ov);
        _post_ir.infer();

        auto recon_ov = _post_ir.get_tensor("recon");

        //TODO: I think we can remove this clone.
        return wrap_ov_tensor_as_torch(recon_ov).clone();
    }
}
