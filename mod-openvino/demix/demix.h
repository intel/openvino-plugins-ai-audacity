// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <openvino/openvino.hpp>
#include <torch/torch.h>

namespace ov_demix
{
    class DemixModel
    {
    public:

        virtual torch::Tensor run(torch::Tensor mix) = 0;

        virtual int64_t chunk_size() = 0;
        virtual int64_t num_instruments() = 0;
        virtual bool PadOuterAndCrossFade() { return true; };

        enum class PadMode
        {
            Constant0,
            Reflect
        };
        virtual PadMode pad_mode() { return PadMode::Constant0; };
    };

    typedef bool (*ProgressUpdate)(double perc_complete, void* user);

    typedef std::pair<std::shared_ptr<std::vector<float>>, std::shared_ptr <std::vector<float>>> AudioTrack;

    // This function is a C++ port of the 'demix' function found within ZFTurbo's excellent 'Music-Source-Separation-Training' project.
    // Here's a link to the implementation that was ported: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/a8a862231de29e6832a26c590e5c29f82f736f66/utils/model_utils.py#L15
    // Note that there are some minor tweaks...
    std::vector<AudioTrack> Demix(std::shared_ptr<DemixModel> model,
        const AudioTrack& track,
        const int num_overlap = 2,
        ProgressUpdate fn = nullptr,
        void* progress_update_user = nullptr);


    // Helper (inplace) function to generate an instrumental track given
    // 1. A stem track (i.e. vocals, drums, etc.)
    // 2. The original input track
    // This function simply computes (input_output - stem_track), and will
    // save the results back into input_output (i.e. inplace processing).
    void GenerateInstrumental(const AudioTrack& stem_track, AudioTrack& input_output);
}
