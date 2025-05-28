// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "musicgen_decoder.h"

namespace ov_musicgen
{
   class StaticKVCacheManagerCL : public StaticKVCacheManager
   {
   public:

      StaticKVCacheManagerCL(ov::Core &core, ov::InferRequest infer_request_initial,
         ov::InferRequest infer_request_with_past,
         ov::InferRequest infer_request_without_past,
         const MusicgenDecoder::Config& decoder_config);

      virtual void Init() override;

      virtual void Reset() override;

      virtual void UpdateFromSingle(size_t position) override;
      virtual void UpdateFromLargeContext() override;

   private:

      class Impl;
      std::shared_ptr< Impl > _impl;

   };


}
