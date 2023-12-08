// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "effects/StatefulEffect.h"
#include <wx/weakref.h>

class WaveTrack;
class wxChoice;

class EffectOVMusicSeparation final : public StatefulEffect
{
   public:

      static const ComponentInterfaceSymbol Symbol;

      EffectOVMusicSeparation();
      virtual ~EffectOVMusicSeparation();

      // ComponentInterface implementation
      ComponentInterfaceSymbol GetSymbol() const override;
      TranslatableString GetDescription() const override;
      VendorSymbol GetVendor() const override;

      unsigned GetAudioInCount() const override;

      // EffectDefinitionInterface implementation

      EffectType GetType() const override;
      bool IsInteractive() const override;

      bool Process(EffectInstance & instance, EffectSettings & settings) override;

      bool UpdateProgress(double perc);

      std::unique_ptr<EffectEditor> PopulateOrExchange(
         ShuttleGui& S, EffectInstance& instance,
         EffectSettingsAccess& access, const EffectOutputs* pOutputs) override;
      bool TransferDataToWindow(const EffectSettings& settings) override;

   protected:

      wxChoice* mTypeChoiceDeviceCtrl;
      int m_deviceSelectionChoice = 0;

    private:

       enum control
       {
          ID_Type = 10000,
       };

       std::vector< std::string > mSupportedDevices;
       std::vector< EnumValueSymbol > mGuiDeviceSelections;

       wxChoice* mTypeChoiceSeparationModeCtrl;
       int m_separationModeSelectionChoice = 0;
       std::vector< EnumValueSymbol > mGuiSeparationModeSelections;

       wxWeakRef<wxWindow> mUIParent{};

};

