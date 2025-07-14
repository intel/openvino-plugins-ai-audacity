// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "OVDemixerEffect.h"

class EffectOVMusicRestoration final : public EffectOVDemixerEffect
{
   public:
      static const ComponentInterfaceSymbol Symbol;

      EffectOVMusicRestoration();
      virtual ~EffectOVMusicRestoration();

      // ComponentInterface implementation
      ComponentInterfaceSymbol GetSymbol() const override;
      TranslatableString GetDescription() const override;

protected:

      const std::string ModelManagerName() const override;
      std::unordered_map< std::string, SeparationModeEntry > GetModelMap() override;
};

