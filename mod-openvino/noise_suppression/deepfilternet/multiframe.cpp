// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#include "multiframe.h"

namespace ov_deepfilternet
{
   MultiFrameModule::MultiFrameModule(int64_t num_freqs, int64_t frame_size, int64_t lookahead, bool real)
   {
      _num_freqs = num_freqs;
      _frame_size = frame_size;
      _real = real;

      if (real)
      {
         _pad3d = torch::nn::ConstantPad3d(torch::nn::ConstantPad3dOptions({ 0, 0, 0, 0, frame_size - 1 - lookahead, lookahead }, 0.0));
      }
      else
      {
         _pad2d = torch::nn::ConstantPad2d(torch::nn::ConstantPad2dOptions({ 0, 0, frame_size - 1 - lookahead, lookahead }, 0.0));
      }

      _need_unfold = (frame_size > 1);
      _lookahead = lookahead;
   }

   torch::Tensor MultiFrameModule::spec_unfold(torch::Tensor spec)
   {
      if (_need_unfold)
      {
         if (_real)
         {
            return _pad3d(spec).unfold(2, _frame_size, 1);
         }
         else
         {
            return _pad2d(spec).unfold(2, _frame_size, 1);
         }

      }
      return spec.unsqueeze(-1);
   }

   static torch::Tensor df(torch::Tensor spec, torch::Tensor coefs)
   {
      return torch::einsum("...tfn,...ntf->...tf", { spec, coefs });
   }

   DF::DF(int64_t num_freqs, int64_t frame_size, int64_t lookahead, bool conj)
      : MultiFrameModule(num_freqs, frame_size, lookahead), _conj(conj)
   {

   }

   torch::Tensor DF::forward(torch::Tensor spec, torch::Tensor coefs)
   {
      auto spec_u = spec_unfold(torch::view_as_complex(spec));
      coefs = torch::view_as_complex(coefs);
      auto spec_f = spec_u.narrow(-2, 0, _num_freqs);
      std::vector< int64_t > view_shape = { coefs.size(0), -1, _frame_size };
      for (size_t i = 2; i < coefs.sizes().size(); i++)
         view_shape.push_back(coefs.size(i));

      coefs = coefs.view(view_shape);

      if (_conj)
      {
         coefs = coefs.conj();
      }

      spec_f = df(spec_f, coefs);

      using namespace torch::indexing;
      spec.index_put_({ "...", Slice(None, _num_freqs), Slice(None) }, torch::view_as_real(spec_f));
      return spec;
   }
}
