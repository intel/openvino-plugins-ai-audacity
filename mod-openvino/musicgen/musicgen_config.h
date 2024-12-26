// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <string>
#include <optional>

namespace ov_musicgen
{
   struct MusicGenConfig
   {
      // When continuing audio, either from a pre-existing wav, or when we are generating
      // music beyond the max token size of the decoder model, we will use this context
      // size to predict the next N seconds of audio from.
      enum class ContinuationContext
      {
         FIVE_SECONDS,
         TEN_SECONDS
      };

      ContinuationContext m_continuation_context = ContinuationContext::FIVE_SECONDS;

      enum class ModelSelection
      {
         MUSICGEN_SMALL_FP16,
         MUSICGEN_SMALL_INT8,
         MUSICGEN_MEDIUM_FP16,
         MUSICGEN_MEDIUM_INT8,
      };

      ModelSelection model_selection = ModelSelection::MUSICGEN_SMALL_FP16;

      //device used to convert wav to id's
      std::string encodec_enc_device = "CPU";

      //device used to convert id's to wav
      std::string encodec_dec_device = "CPU";

      //the two devices used to run inference to predict each next token.
      std::string musicgen_decode_device0 = "CPU";
      std::string musicgen_decode_device1 = "CPU";

      //the folder containing all the models
      std::string model_folder;

      //use stereo models, produce stereo output.
      bool bStereo = false;

      //folder to cache compiled openvino models into.
      std::optional< std::string > cache_folder = std::optional< std::string >{};

      std::optional< std::string > performance_hint = std::optional< std::string >{};
   };

   struct CallbackParams
   {
      typedef bool (*CallbackFunc)(float perc_complete, //<- range 0 to 1
         void* user);

      CallbackFunc callback;

      // Call 'CallbackFunc' for every n new tokens generated.
      // Remeber that 50 tokens = 1 second of audio. So 5 would be
      // a callback for each 0.1 second of new audio generated.
      size_t every_n_new_tokens;
      void* user;
   };

}
