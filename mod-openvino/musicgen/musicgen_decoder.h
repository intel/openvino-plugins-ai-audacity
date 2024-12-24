// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <torch/torch.h>
#include <openvino/openvino.hpp>
#include <optional>
#include "musicgen_config.h"

namespace ov_musicgen
{
    class MusicgenDecoder
    {
    public:

        virtual void Reset() = 0;

        virtual torch::Tensor run(torch::Tensor input_ids,
            std::optional<torch::Tensor> encoder_hidden_states,
            std::optional<torch::Tensor> encoder_attention_mask) = 0;

        virtual int64_t PastLength() = 0;

        virtual int64_t MaxNewTokens() = 0;
    };

    class MusicgenDecoderStatic : public MusicgenDecoder
    {
    public:

        MusicgenDecoderStatic(ov::Core& core, MusicGenConfig& config);

        virtual void Reset() override;

        virtual torch::Tensor run(torch::Tensor input_ids,
            std::optional<torch::Tensor> encoder_hidden_states,
            std::optional<torch::Tensor> encoder_attention_mask) override;

        virtual int64_t PastLength() override;

        virtual int64_t MaxNewTokens() override;

    private:

        int64_t _past_length = 0;

        ov::InferRequest _infer_request;
        ov::InferRequest _infer_request_initial;
        ov::InferRequest _infer_request_nonkv;

        std::vector< ov::Tensor > past_decoder_keys;
        std::vector< ov::Tensor > past_decoder_values;
        std::vector< ov::Tensor > past_encoder_keys;
        std::vector< ov::Tensor > past_encoder_values;

        std::vector< ov::Tensor > present_decoder_keys;
        std::vector< ov::Tensor > present_decoder_values;

        std::vector< ov::Tensor > present_decoder_keys_large_context;
        std::vector< ov::Tensor > present_decoder_values_large_context;
    };
}
