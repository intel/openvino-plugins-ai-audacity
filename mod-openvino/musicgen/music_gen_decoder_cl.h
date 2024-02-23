#pragma once

#include <openvino/opsets/opset8.hpp>
#include "musicgen_decoder_model.h"
#include "musicgen_utils.h"
#include "musicgen_config.h"

namespace ov_musicgen
{
   class MusicgenDecoderModelCL : public MusicgenDecoderModel
   {
   public:

      const size_t N_LAYERS = 24;

      MusicgenDecoderModelCL(ov::Core& core, MusicGenConfig& config);

      virtual int64_t MaxNewTokens() override;

      virtual void Reset() override;

      virtual void ShiftLeft(int64_t ntokens) override;

      virtual ov::Tensor run(torch::Tensor hidden_states, std::optional<torch::Tensor> encoder_hidden_states, torch::Tensor encoder_attention_mask) override;

      virtual ov::Tensor get_last_hidden_state() override;

      virtual int64_t PastLength() override;

   private:

      ov::InferRequest _infer_request;

      //input tensors (OpenVINO tensors & OV Tensors wrapped as torch::Tensor's)
      torch::Tensor _hidden_states;
      ov::Tensor _encoder_attention_mask_ov;
      torch::Tensor _encoder_attention_mask;
      ov::Tensor _custom_attention_mask;

      std::vector< std::vector< ov::Tensor > > _past_key_values_ov;

      //output tensors
      ov::Tensor _last_hidden_state_ov;
      torch::Tensor _last_hidden_state;  //simply a wrapper around _last_hidden_state_ov (pointing to same underlying buffer)
      std::vector< std::vector< ov::Tensor > > _new_key_values_ov;

      int64_t _past_length = 0;

      //model that needs to run when _past_length is 0 to produce initial key/vals
      ov::InferRequest _infer_request_initial;
      torch::Tensor _intiial_encoder_hidden_state;
      std::vector< std::vector< torch::Tensor > > _initial_past_key_values;


      //todo, this should go away:
      torch::Tensor _attention_mask;


      //large context stuff
      ov::InferRequest _infer_request_large_context;

      struct CLStuff;
      std::shared_ptr< CLStuff > _cl_stuff;
   };
}
