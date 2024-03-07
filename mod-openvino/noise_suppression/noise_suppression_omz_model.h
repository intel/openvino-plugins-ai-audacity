// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <openvino/openvino.hpp>

#include "noise_suppression_model.h"

class NoiseSuppressionOMZModel : public NoiseSuppressionModel
{
public:

   NoiseSuppressionOMZModel(std::string model_path, std::string device, std::string cache_dir);

   virtual int sample_rate() override
   {
      return _freq_model;
   }

   virtual bool run(std::shared_ptr<WaveChannel> pChannel, sampleCount start, size_t total_samples,
      ProgressCallbackFunc callback = nullptr, void* callback_user = nullptr) override;

private:

   void _compile_noise_suppression_model(std::string model_path, std::string device, std::string cache_dir);

   ov::CompiledModel _compiledModel;
   ov::InferRequest _infer_request;

   int _freq_model;
};
