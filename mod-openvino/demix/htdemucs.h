// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <demix/demix.h>

namespace ov_demix
{
    class HTDemucs : public DemixModel
    {
    public:
        HTDemucs(const std::string& model_dir, const std::string& device, const std::string& cache_dir = "");

        torch::Tensor run(torch::Tensor mix) override;
        int64_t chunk_size() override { return _chunk_size; }
        int64_t num_instruments() override { return _num_instruments; }
        bool PadOuterAndCrossFade() override { return false; };

    private:
        ov::InferRequest _forward_ir;
        int64_t _chunk_size = 343980;
        int64_t _num_instruments = 4;
    };
}