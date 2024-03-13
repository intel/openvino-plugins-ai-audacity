// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <torch/torch.h>

namespace ov_deepfilternet
{
   class MultiFrameModule
   {
   public:

      MultiFrameModule(int64_t num_freqs, int64_t frame_size, int64_t lookahead = 0, bool real = false);

      virtual torch::Tensor forward(torch::Tensor spec, torch::Tensor coefs) = 0;

   protected:

      torch::Tensor spec_unfold(torch::Tensor spec);

      int64_t _num_freqs;
      int64_t _frame_size;
      bool _real;
      bool _need_unfold;
      int64_t _lookahead;

      torch::nn::ConstantPad3d _pad3d{ nullptr }; // For 3D padding
      torch::nn::ConstantPad2d _pad2d{ nullptr }; // For 2D padding
   };

   class DF : public MultiFrameModule
   {
   public:

      DF(int64_t num_freqs, int64_t frame_size, int64_t lookahead = 0, bool conj = false);

      virtual torch::Tensor forward(torch::Tensor spec, torch::Tensor coefs) override;

   private:

      bool _conj;
   };
}
