#pragma once

#include <torch/torch.h>
#include <openvino/openvino.hpp>
#include <optional>

class MusicgenDecoderModel
{
public:

    virtual void Reset() = 0;

    virtual ov::Tensor run(torch::Tensor hidden_states, 
        std::optional<torch::Tensor> encoder_hidden_states, 
        torch::Tensor encoder_attention_mask) = 0;

    virtual ov::Tensor get_last_hidden_state() = 0;

    virtual int64_t PastLength() = 0;

    virtual int64_t MaxNewTokens() = 0;

    virtual void ShiftLeft(int64_t ntokens) = 0;
};

