// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <torch/torch.h>
#include <openvino/openvino.hpp>
#include "musicgen_utils.h"
#include "musicgen_config.h"

namespace ov_musicgen
{
   class MusicGenEncodecEncoder
   {
   public:

      MusicGenEncodecEncoder(ov::Core& core, MusicGenConfig& config)
      {
         auto model_folder = config.model_folder;
         std::string modelpath;
         switch (config.m_continuation_context)
         {
         case MusicGenConfig::ContinuationContext::FIVE_SECONDS:
         {
            modelpath = FullPath(model_folder, "encodec_encoder_5s.xml");
         }
         break;

         case MusicGenConfig::ContinuationContext::TEN_SECONDS:
         {
            modelpath = FullPath(model_folder, "encodec_encoder_10s.xml");
         }
         break;
         }

         auto binfile = FullPath(model_folder, "encodec_encoder_combined_weights.bin");

         std::shared_ptr<ov::Model> model = core.read_model(modelpath, binfile);
         logBasicModelInfo(model);

         ov::CompiledModel compiledModel = core.compile_model(model, config.encodec_enc_device);

         _infer_request = compiledModel.create_infer_request();
         _infer_request.infer(); //warm up run.
      }

      torch::Tensor encode(torch::Tensor input_values)
      {
         auto input_ov_wrapped = wrap_ov_tensor_as_torch(_infer_request.get_input_tensor());

         input_ov_wrapped.copy_(input_values);

         _infer_request.infer();

         auto audio_codes = wrap_ov_tensor_as_torch(_infer_request.get_output_tensor());

         return audio_codes;
      }

   private:

      ov::InferRequest _infer_request;
   };
}
