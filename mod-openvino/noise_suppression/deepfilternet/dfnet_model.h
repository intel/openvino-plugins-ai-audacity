// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <optional>
#include <openvino/openvino.hpp>
#include "multiframe.h"

namespace ov_deepfilternet
{
   enum class ModelSelection
   {
      DEEPFILTERNET2,
      DEEPFILTERNET3,
   };

   class Mask;
   class DFNetModel
   {
   public:

      DFNetModel(std::string model_folder,
         std::string device,
         ModelSelection model_selection,
         std::optional<std::string> openvino_cache_dir,
         torch::Tensor erb_widths,
         int64_t lookahead = 2, int64_t nb_df = 96);

      torch::Tensor
         forward(torch::Tensor spec, torch::Tensor feat_erb, torch::Tensor feat_spec, bool post_filter=false);

      int64_t num_static_hops()
      {
         return _num_hops;
      };

   private:

      torch::Tensor
         forward_df3(torch::Tensor spec, torch::Tensor feat_erb, torch::Tensor feat_spec, bool post_filter);

      torch::Tensor
         forward_df2(torch::Tensor spec, torch::Tensor feat_erb, torch::Tensor feat_spec);

      ov::InferRequest _infer_request_enc;
      ov::InferRequest _infer_request_erb_dec;
      ov::InferRequest _infer_request_df_dec;
      std::shared_ptr< ov::Core > _core;

      std::shared_ptr< torch::nn::ConstantPad3d > _pad_spec;
      std::shared_ptr< torch::nn::ConstantPad2d > _pad_feat;
      std::shared_ptr< Mask > _mask;

      int64_t _nb_df;
      DF _df;

      int64_t _num_hops;

      bool _bDF3;
   };
}

