// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "OVDemixerEffect.h"

class EffectOVReverbRemoval final : public EffectOVDemixerEffect
{
   public:
      static const ComponentInterfaceSymbol Symbol;

      EffectOVReverbRemoval();
      virtual ~EffectOVReverbRemoval();

      // ComponentInterface implementation
      ComponentInterfaceSymbol GetSymbol() const override;
      TranslatableString GetDescription() const override;

protected:

      const std::string ModelManagerName() const override;
      std::unordered_map< std::string, SeparationModeEntry > GetModelMap() override;
};

