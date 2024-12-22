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
   class MusicgenForCausalLM;
   class MusicGenEncodecEncoder;
   class MusicgenDecoder;

   class MusicgenForConditionalGenerationRefactor
   {
   public:

      MusicgenForConditionalGenerationRefactor(MusicGenConfig& config);

      struct BaseModelOutput
      {
         // Sequence of hidden-states at the output of the last layer of the model.
         std::optional<torch::Tensor> last_hidden_state = {};

         // Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
         std::optional<torch::Tensor> hidden_states = {};

         //Attentions weights after the attention softmax, used to compute the weighted average in the self - attention heads.
         std::optional<torch::Tensor> attentions = {};
      };

      struct GenerateReturn
      {
         std::shared_ptr<std::vector<float>> wav;
         std::shared_ptr<std::vector<float>> wav1;
         torch::Tensor input_ids;
      };

      struct CallbackTracking
      {
         size_t total_tokens_generated_so_far;
         size_t total_tokens_to_generate;
      };

      GenerateReturn generate(std::optional < torch::Tensor > inputs_tensor,
         int64_t max_token_length,
         std::optional < torch::Tensor > attention_mask,
         CallbackTracking& tracking,
         std::optional< torch::Tensor > audio_to_continue = {},
         std::optional< torch::Tensor > input_ids_to_continue = {},
         float guidance_scale = 3.f,
         int64_t top_k = 250,
         std::optional< CallbackParams > callback_params = {});

      int64_t MaxNewTokens();

      void SetSeed(unsigned int seed);

      std::shared_ptr<std::vector<float>> ids_to_wav(torch::Tensor ids);

   private:

      torch::Tensor apply_delay_pattern_mask(torch::Tensor input_ids, torch::Tensor decoder_pad_token_mask);

      torch::Tensor prepare_inputs_for_generation(torch::Tensor decoder_input_ids,
         torch::Tensor decoder_delay_pattern_mask,
         std::optional<float> guidance_scale);

      //todo: change to operator?
      std::pair<torch::Tensor, torch::Tensor> forward(std::optional<torch::Tensor> input_ids,
         std::optional<torch::Tensor> attention_mask,
         std::optional<torch::Tensor> input_values,
         std::optional<torch::Tensor> padding_mask,
         std::optional<torch::Tensor> decoder_input_ids,
         std::optional< BaseModelOutput > encoder_outputs,
         std::optional< torch::Tensor > encoder_hidden_states_in = {});

      //transformers\generation\utils.py: GenerationMixin: sample
      torch::Tensor sample(torch::Tensor input_ids, std::optional < torch::Tensor > attention_mask,
         torch::Tensor decoder_delay_pattern_mask,
         std::optional< BaseModelOutput > encoder_outputs,
         size_t max_length,
         float guidance_scale,
         int64_t top_k,
         CallbackTracking& tracking,
         std::optional< CallbackParams > callback_params);

      void ShiftLeft(int64_t ntokens);

      torch::Tensor _enc_to_dec_proj(torch::Tensor encoder_hidden_states);

      //transformers\generation\logits_process.py, ClassifierFreeGuidanceLogitsProcessor.
      torch::Tensor _logits_processor(torch::Tensor input_ids, torch::Tensor next_token_logits, float guidance_scale);

      //transformers\generation\logits_process.py, TopKLogitsWarper.
      torch::Tensor _logits_warper(torch::Tensor input_ids,
         torch::Tensor next_token_scores, int64_t top_k, float filter_value);

      std::shared_ptr< MusicgenDecoder > _decoder_refactor;

      std::shared_ptr< MusicGenEncodecEncoder > _encoder;

      ov::InferRequest _enc_to_dec_proj_infer_request;

      int _nforward_calls = 1;

      ov::InferRequest _encodec_infer_request;

      ov::InferRequest _text_encoder_infer_request;

      torch::Generator _generator;

      std::shared_ptr<ov::Core> _core;

      MusicGenConfig _config;

      //returns { input_ids, delayed_pattern_mask }
      std::pair< torch::Tensor, torch::Tensor> _build_delay_pattern_mask(torch::Tensor input_ids, int64_t pad_token_id, int64_t max_length);
   };
}
