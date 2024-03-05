// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <iostream>
#include <openvino/openvino.hpp>
#include <chrono>
#include <optional>
#include <torch/torch.h>
#include <fstream>
#include <ittutils.h>
#include "musicgen_utils.h"
#include "musicgen_config.h"

namespace ov_musicgen
{
   class MusicgenModelStatic;
   class MusicgenForCausalLM
   {
   public:
      MusicgenForCausalLM(ov::Core& core, MusicGenConfig& config);

      int64_t PastLength();
      void Reset();
      void ShiftLeft(int64_t ntokens);
      int64_t MaxNewTokens();

      int64_t NumCodebooks();
      int64_t AudioChannels() { return _audio_channels; };

      //returns logits tensor
      torch::Tensor forward(torch::Tensor input_ids,
         std::optional< torch::Tensor > attention_mask,
         std::optional< torch::Tensor > encoder_hidden_states,
         std::optional< torch::Tensor > encoder_attention_mask,
         std::optional< torch::Tensor > head_mask,
         std::optional< torch::Tensor > cross_attn_head_mask,
         std::optional< torch::Tensor > inputs_embeds
      );

      //returns { input_ids, delayed_pattern_mask }
      std::pair< torch::Tensor, torch::Tensor> build_delay_pattern_mask(torch::Tensor input_ids, int64_t pad_token_id, int64_t max_length);

   private:
      std::shared_ptr< MusicgenModelStatic > _decoder;
      ov::InferRequest _lm_heads_infer_request;
      int _nforward_calls = 1;
      int64_t _audio_channels = 1;
   };
}
