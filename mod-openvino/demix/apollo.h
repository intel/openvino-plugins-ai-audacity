// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <demix/demix.h>

namespace ov_demix
{
    class Apollo : public DemixModel
    {
    public:

        Apollo(const std::string& model_dir,
            const std::string& device,
            const std::string& cache_dir = "");

        torch::Tensor run(torch::Tensor arr) override;
        int64_t chunk_size() override { return _chunk_len; }
        int64_t num_instruments() override { return 1; }

        PadMode pad_mode() override { return ov_demix::DemixModel::PadMode::Reflect; }

    private:

        ov::InferRequest _forward_ir;

        int64_t _hop_len;
        int64_t _win_len;
        torch::Tensor _window;
        int64_t _chunk_len;
    };
}