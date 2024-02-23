#pragma once
#include <optional>
#include <vector>
#include <string>
#include <memory>
#include "musicgen_config.h"

namespace ov_musicgen
{
   class MusicGen
   {
   public:

      MusicGen(MusicGenConfig& config);

      struct AudioContinuationParams
      {
         std::pair<std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>>> audio_to_continue;
         bool bReturnAudioToContinueInOutput = true;
      };

      std::pair<std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>>> Generate(std::optional<std::string> prompt,
         std::optional<AudioContinuationParams> audio_to_continue_params,
         float total_desired_length_seconds,
         std::optional< unsigned int > seed,
         float guidance_scale = 3.f,
         int top_k = 250,
         std::optional< CallbackParams > callback_params = {});

   private:

      struct Impl;
      std::shared_ptr<Impl> _impl;

      MusicGenConfig _config;
   };
}
