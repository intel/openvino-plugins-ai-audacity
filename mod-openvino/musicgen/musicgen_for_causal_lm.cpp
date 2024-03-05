// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#include "musicgen_for_causal_lm.h"
#include "musicgen_model_static.h"

namespace ov_musicgen
{
   MusicgenForCausalLM::MusicgenForCausalLM(ov::Core& core, MusicGenConfig& config)
   {
      _decoder = std::make_shared< MusicgenModelStatic >(core, config);

      //load lm_heads IR
      std::string model_folder = config.model_folder;
      if (config.bStereo)
      {
         model_folder = FullPath(model_folder, "stereo");
         _audio_channels = 2;
      }
      else
      {
         model_folder = FullPath(model_folder, "mono");
         _audio_channels = 1;
      }

      auto modelpath = FullPath(model_folder, "lm_heads.xml");

      std::shared_ptr<ov::Model> model = core.read_model(modelpath);

      model->reshape({ {2, ov::Dimension(), 1024} });

      //If device0 or device1 are set to a GPU device, run the lm_heads model on GPU as well.
      bool bIsDev0GPU = config.musicgen_decode_device0.find("GPU") != std::string::npos;
      bool bIsDev1GPU = config.musicgen_decode_device1.find("GPU") != std::string::npos;

      std::string device = (bIsDev0GPU || bIsDev1GPU) ? (bIsDev0GPU ? config.musicgen_decode_device0 : config.musicgen_decode_device1) : "CPU";

      std::cout << "Compiling lm heads model for device=" << device << std::endl;
      auto lm_heads_compiled_model = core.compile_model(model, device);

      _lm_heads_infer_request = lm_heads_compiled_model.create_infer_request();

      //link together the output of decoder model & lm_heads model.
      _lm_heads_infer_request.set_input_tensor(_decoder->get_last_hidden_state());
      _lm_heads_infer_request.infer();
   }

   int64_t MusicgenForCausalLM::PastLength()
   {
      return _decoder->PastLength();
   };

   void MusicgenForCausalLM::Reset()
   {
      _decoder->Reset();
   }

   void MusicgenForCausalLM::ShiftLeft(int64_t ntokens)
   {
      _decoder->ShiftLeft(ntokens);
   }

   int64_t MusicgenForCausalLM::NumCodebooks()
   {
      return _decoder->NumCodebooks();
   }

   int64_t MusicgenForCausalLM::MaxNewTokens()
   {
      return _decoder->MaxNewTokens();
   }

   torch::Tensor MusicgenForCausalLM::forward(torch::Tensor input_ids,
      std::optional< torch::Tensor > attention_mask,
      std::optional< torch::Tensor > encoder_hidden_states,
      std::optional< torch::Tensor > encoder_attention_mask,
      std::optional< torch::Tensor > head_mask,
      std::optional< torch::Tensor > cross_attn_head_mask,
      std::optional< torch::Tensor > inputs_embeds
   )
   {
      ITT_SCOPED_TASK(MusicgenForCausalLM_forward)

         auto hidden_states_ov = _decoder->forward(input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            head_mask,
            cross_attn_head_mask,
            inputs_embeds);

      _lm_heads_infer_request.set_input_tensor(hidden_states_ov);

      {
         ITT_SCOPED_TASK(lm_heads_infer)
            _lm_heads_infer_request.infer();
      }

      auto logits_ov = _lm_heads_infer_request.get_output_tensor();
      auto logits = wrap_ov_tensor_as_torch(logits_ov);

      logits = logits.reshape({ -1, logits.size(2),  logits.size(3) });

      _nforward_calls++;

      return logits;
   }

   //returns { input_ids, delayed_pattern_mask }
   std::pair< torch::Tensor, torch::Tensor> MusicgenForCausalLM::build_delay_pattern_mask(torch::Tensor input_ids, int64_t pad_token_id, int64_t max_length)
   {
      using namespace torch::indexing;

      // (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
      input_ids = input_ids.reshape({ -1, _decoder->NumCodebooks(), input_ids.sizes().back() });
      auto bsz = input_ids.sizes()[0];
      auto num_codebooks = input_ids.sizes()[1];
      auto seq_len = input_ids.sizes()[2];

      auto input_ids_shifted = (
         torch::ones({ bsz, num_codebooks, max_length }, torch::TensorOptions().dtype(torch::kInt64)) * -1
         );

      int64_t channel_codebooks;
      if (_audio_channels == 2)
      {
         channel_codebooks = num_codebooks / 2;
      }
      else
      {
         channel_codebooks = num_codebooks;
      }

      // we only apply the mask if we have a large enough seq len - otherwise we return as is
      if (max_length < (2 * channel_codebooks - 1))
      {
         return { input_ids.reshape({bsz * num_codebooks, -1}), input_ids_shifted.reshape({bsz * num_codebooks, -1}) };
      }

      // fill the shifted ids with the prompt entries, offset by the codebook idx
      for (int64_t codebook = 0; codebook < channel_codebooks; codebook++)
      {
         if (_audio_channels == 1)
         {
            // mono channel - loop over the codebooks one-by-one
            input_ids_shifted.index_put_({ Slice(), codebook, Slice(codebook, seq_len + codebook) }, input_ids.index({ Slice(), codebook }));
         }
         else
         {
            // left/right channels are interleaved in the generated codebooks, so handle one then the other
            input_ids_shifted.index_put_({ Slice(), 2 * codebook, Slice(codebook, seq_len + codebook) }, input_ids.index({ Slice(), 2 * codebook }));
            input_ids_shifted.index_put_({ Slice(), 2 * codebook + 1, Slice(codebook, seq_len + codebook) }, input_ids.index({ Slice(), 2 * codebook + 1 }));
         }
      }

      // construct a pattern mask that indicates the positions of padding tokens for each codebook
      // first fill the upper triangular part (the EOS padding)
      auto delay_pattern = torch::triu(
         torch::ones({ channel_codebooks, max_length }, torch::TensorOptions().dtype(torch::kBool)), max_length - channel_codebooks + 1
      );

      // then fill the lower triangular part (the BOS padding)
      delay_pattern = delay_pattern + torch::tril(torch::ones({ channel_codebooks, max_length }, torch::TensorOptions().dtype(torch::kBool)));

      if (_audio_channels == 2)
      {
         // for left/right channel we need to duplicate every row of the pattern mask in an interleaved fashion
         delay_pattern = delay_pattern.repeat_interleave(2, 0);
      }

      auto mask = ~delay_pattern;
      input_ids = mask * input_ids_shifted + ~mask * pad_token_id;

      //dump_tensor(input_ids, "ov_input_ids_after_mask_op.raw");

      // find the first position to start generating - this is the first place we have the -1 token
      // and will always be in the first codebook (since it has no codebook offset)
      auto first_codebook_ids = input_ids.index({ Slice(), 0, Slice() });
      auto start_ids = (first_codebook_ids == -1).nonzero().index({ Slice(), 1 });

      int64_t first_start_id;
      if (start_ids.numel() > 0)
      {
         first_start_id = start_ids.min().item().toLong();
      }
      else
      {
         //we have no tokens that need to be filled - return entire matrix of input ids
         first_start_id = seq_len;
      }

      // (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
      auto pattern_mask = input_ids.reshape({ bsz * num_codebooks, -1 });

      input_ids = input_ids.index({ "...", Slice(None, first_start_id) }).reshape({ bsz * num_codebooks, -1 });

      return { input_ids , pattern_mask };
   }
}
