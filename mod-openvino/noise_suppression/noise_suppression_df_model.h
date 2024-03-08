// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "noise_suppression_model.h"
#include "deepfilternet/deepfilter.h"

class NoiseSuppressionDFModel : public NoiseSuppressionModel
{
public:

   NoiseSuppressionDFModel(std::string model_path, std::string device,
      std::string cache_dir, ov_deepfilternet::ModelSelection model_selection);

   virtual int sample_rate() override
   {
      return 48000;
   }

   void SetAttenLimit(float atten_limit)
   {
      _atten_limit = atten_limit;
   }

   void SetDF3PostFilter(float post_filter)
   {
      _bDF3_post_filter = post_filter;
   }

   virtual bool run(std::shared_ptr<WaveChannel> pChannel, sampleCount start, size_t total_samples,
      ProgressCallbackFunc callback = nullptr, void* callback_user = nullptr) override;

private:

   std::shared_ptr<ov_deepfilternet::DeepFilter> _df;
   float _atten_limit = 100.f;

   bool _bDF3_post_filter = false;

};
