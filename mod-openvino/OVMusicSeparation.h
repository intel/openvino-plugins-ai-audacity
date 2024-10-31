// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "StatefulEffect.h"
#include "effects/StatefulEffectUIServices.h"
#include <wx/weakref.h>

class WaveTrack;
class wxChoice;
class wxCheckBox;
class wxTextCtrl;
class wxSizer;

class EffectOVMusicSeparation final : public StatefulEffect, public StatefulEffectUIServices
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

      void OnAdvancedCheckboxChanged(wxCommandEvent& evt);
      void OnDeviceInfoButtonClicked(wxCommandEvent& evt);

   protected:

      wxChoice* mTypeChoiceDeviceCtrl;
      int m_deviceSelectionChoice = 0;

    private:

       enum control
       {
          ID_Type = 10000,
          ID_Type_AdvancedCheckbox,
          ID_Type_DeviceInfoButton
       };

       std::vector< std::string > mSupportedDevices;
       std::vector< EnumValueSymbol > mGuiDeviceSelections;

       wxChoice* mTypeChoiceSeparationModeCtrl;
       int m_separationModeSelectionChoice = 0;
       std::vector< EnumValueSymbol > mGuiSeparationModeSelections;

       wxWeakRef<wxWindow> mUIParent{};

       wxCheckBox* mShowAdvancedOptionsCheckbox;

       int mNumberOfShifts = 1;
       wxTextCtrl* mNumberOfShiftsCtrl = nullptr;

       void show_or_hide_advanced_options();
       wxSizer* advancedSizer = nullptr;
       bool mbAdvanced = false;

       std::vector<std::pair<std::string, std::string>> m_simple_to_full_device_map;

       DECLARE_EVENT_TABLE()
};

