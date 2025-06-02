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

         auto modelpath = FullPath(model_folder, "openvino_encodec_encode.xml");

         // there seems to be some issue in using the same ov::Core that was used for
         // encodec_decoder.. GPU team investigating. So, as a workaround, instantiate
         // a new core and use it instead.
         //TODO: Remove this when the issue is resolved.
         ov::Core wa_core;
         if (config.cache_folder)
         {
            wa_core.set_property(ov::cache_dir(*config.cache_folder));
         }

         std::shared_ptr<ov::Model> model = wa_core.read_model(modelpath);

         switch (config.m_continuation_context)
         {
            case MusicGenConfig::ContinuationContext::FIVE_SECONDS:
            {
               model->reshape({ 1, 1, 5 * 32000 });
            }
            break;

            case MusicGenConfig::ContinuationContext::TEN_SECONDS:
            {
               model->reshape({ 1, 1, 10 * 32000 });
            }
            break;
         }
         std::cout << "openvino_encodec_encode:" << std::endl;
         logBasicModelInfo(model);
         ov::CompiledModel compiledModel = wa_core.compile_model(model, config.encodec_enc_device);
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
