#pragma once

#include <iostream>
#include <openvino/openvino.hpp>

#include <chrono>
#include <optional>
#include <torch/torch.h>
#include <fstream>

#include <ittutils.h>
#include "musicgen_decoder_model.h"

#include "musicgen_utils.h"

#include "musicgen_config.h"


class MusicgenSinusoidalPositionalEmbedding;
class MusicgenModelStatic
{
public:

    const size_t N_LAYERS = 24;
    const size_t MAX_TENSOR_HEIGHT = 1503;

    MusicgenModelStatic(ov::Core& core, MusicGenConfig& config);

    //int64_t PastLength() { return _past_length; };
    int64_t PastLength() { return _decoder_model->PastLength(); };

    int64_t MaxNewTokens()
    {
        return _decoder_model->MaxNewTokens();
    }

    void Reset()
    {
        _decoder_model->Reset();
    }

    void ShiftLeft(int64_t ntokens)
    {
        _decoder_model->ShiftLeft(ntokens);
    }

    int64_t NumCodebooks() { return _num_codebooks; };

    ov::Tensor get_last_hidden_state() { return _decoder_model->get_last_hidden_state(); };

    ov::Tensor forward(std::optional<torch::Tensor> input_ids,
        std::optional<torch::Tensor> attention_mask,
        std::optional<torch::Tensor> encoder_hidden_states,
        std::optional<torch::Tensor> encoder_attention_mask,
        std::optional<torch::Tensor> head_mask,
        std::optional<torch::Tensor> cross_attn_head_mask,
        std::optional<torch::Tensor> inputs_embeds);

private:

    torch::Tensor _prepare_4d_attention_mask(torch::Tensor mask, int64_t tgt_len);

    std::shared_ptr< MusicgenDecoderModel > _decoder_model;

    torch::Tensor _embed_tokens(torch::Tensor input);

    ov::InferRequest _embed_tokens_infer_request;

    std::shared_ptr< MusicgenSinusoidalPositionalEmbedding > _embed_positions;

    uint64_t _total_decode_time = 0;

    int64_t _num_codebooks = 4;
};