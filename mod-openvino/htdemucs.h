// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <memory>
#include <vector>

namespace torch
{
    class Tensor;
}

namespace ovdemucs
{
    struct HTDemucs_impl;
    struct HTDemucs_openvino_impl;
    class TensorChunk;

    //todo: make this bool so that registered callback can return false trigger cancel.
    typedef bool (*ProgressUpdate)(double perc_complete, void *user);
    class HTDemucs
    {
    public:

        HTDemucs(const char *model_path, const std::string &device = "CPU", const std::string cache_dir="");

        //will return true if processing completed, false if processing was cancelled, and
        // will throw an exception upon error.
        bool Apply(float* pIn, int64_t nsamples,
           float* &pOut0,
           float* &pOut1,
           float* &pOut2,
           float* &pOut3,
           ProgressUpdate fn = nullptr,
           void* progress_update_user = nullptr);

        static std::vector<std::string> GetSupportedDevices();

    private:
#ifdef ONNX_SUPPORT
        std::shared_ptr< HTDemucs_impl > _impl;
#endif
        std::shared_ptr< HTDemucs_openvino_impl > _impl_ov;
        bool _apply_model_0(torch::Tensor& mix, torch::Tensor& out, int64_t shifts = 1, bool split = true, double overlap = 0.25, double transition_power = 1., int64_t static_shifts = 1);
        bool _apply_model_1(torch::Tensor& mix, torch::Tensor& out, int64_t shifts = 1, bool split = true, double overlap = 0.25, double transition_power = 1., int64_t static_shifts = 1);
        bool _apply_model_2(TensorChunk& mix, torch::Tensor& out, bool split = true, double overlap = 0.25, double transition_power = 1., int64_t static_shifts = 1);
        bool _apply_model_3(TensorChunk& mix, torch::Tensor& out);
        bool _actually_run_model(torch::Tensor& mix_tensor, torch::Tensor& x);

        int64_t _shifts = 0;
        int64_t _offsets = 0;
        int64_t _inference_i = 0;

        struct Priv;
        std::shared_ptr< Priv > _priv;

    };
}
