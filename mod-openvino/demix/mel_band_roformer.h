// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <demix/demix.h>

namespace ov_demix
{
    class MelBandRoformer : public DemixModel
    {
    public:
        MelBandRoformer(const std::string& model_dir,
            const std::string& device,
            const std::string& cache_dir = "",
            DemixModel::PadMode pad_mode = DemixModel::PadMode::Constant0);

        torch::Tensor run(torch::Tensor arr) override;
        int64_t chunk_size() override { return 352800; }
        int64_t num_instruments() override { return 1; }

        PadMode pad_mode() override { return _pad_mode; }

    private:

        ov::InferRequest _pre_ir;
        ov::InferRequest _forward_ir;
        ov::InferRequest _post_ir;

        DemixModel::PadMode _pad_mode;
    };
}
