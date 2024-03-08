// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#include "noise_suppression_df_model.h"

NoiseSuppressionDFModel::NoiseSuppressionDFModel(std::string model_path, std::string device, std::string cache_dir,
   ov_deepfilternet::ModelSelection model_selection)
{
   _df = std::make_shared< ov_deepfilternet::DeepFilter >(model_path, device, model_selection, cache_dir);
}

bool NoiseSuppressionDFModel::run(std::shared_ptr<WaveChannel> pChannel, sampleCount start, size_t total_samples,
   ProgressCallbackFunc callback, void* callback_user)
{
   bool ret = true;

   Floats entire_input{ total_samples };
   bool bOkay = pChannel->GetFloats(entire_input.get(), start, total_samples);
   if (!bOkay)
   {
      throw std::runtime_error("Unable to get " + std::to_string(total_samples) + " samples.");
   }

   torch::Tensor input_wav_tensor = torch::from_blob(entire_input.get(), { 1, (int64_t)total_samples });


   std::optional<float> atten_lim;
   if (_atten_limit != 100.f)
   {
      atten_lim = _atten_limit;
   }

   auto wav = _df->filter(input_wav_tensor, atten_lim, 20, _bDF3_post_filter, callback, callback_user);

   if (!wav)
   {
      std::cout << "!wav -- returning false" << std::endl;
      return false;
   }

   ret = pChannel->Set((samplePtr)(wav->data()), floatSample, start, total_samples);

   return ret;
}
