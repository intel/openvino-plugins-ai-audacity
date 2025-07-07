// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "OVDemixerEffect.h"
#include <wx/weakref.h>

class EffectOVMusicSeparation final : public EffectOVDemixerEffect
{
   public:

      static const ComponentInterfaceSymbol Symbol;

      EffectOVMusicSeparation();
      virtual ~EffectOVMusicSeparation();

      // ComponentInterface implementation
      ComponentInterfaceSymbol GetSymbol() const override;
      TranslatableString GetDescription() const override;
};

