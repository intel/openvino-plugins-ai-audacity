// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "StatefulEffect.h"
#include "effects/StatefulEffectUIServices.h"
#include "ShuttleAutomation.h"
#include <wx/weakref.h>

class WaveTrack;
class wxChoice;
class WaveChannel;
class wxCheckBox;
class wxSizer;

namespace ov
{
   class CompiledModel;
}

class EffectOVNoiseSuppression final : public StatefulEffect, public StatefulEffectUIServices
{
public:

   static const ComponentInterfaceSymbol Symbol;

   EffectOVNoiseSuppression();
   virtual ~EffectOVNoiseSuppression();

   // ComponentInterface implementation
   ComponentInterfaceSymbol GetSymbol() const override;
   TranslatableString GetDescription() const override;
   VendorSymbol GetVendor() const override;

   // EffectDefinitionInterface implementation

   EffectType GetType() const override;
   bool IsInteractive() const override;
   bool SaveSettingsAsString(
      const EffectSettings& settings, wxString& parms) const override;
   [[nodiscard]] OptionalMessage LoadSettingsFromString(
      const wxString& parms, EffectSettings& settings) const override;
   bool SaveSettings(
      const EffectSettings& settings, CommandParameters& parms) const override;
   bool LoadSettings(
      const CommandParameters& parms, EffectSettings& settings) const override;

   OptionalMessage LoadUserPreset(
      const RegistryPath& name, EffectSettings& settings) const override;
   bool SaveUserPreset(
      const RegistryPath& name, const EffectSettings& settings) const override;

   bool Process(EffectInstance& instance, EffectSettings& settings) override;

   bool UpdateProgress(double perc);

   std::unique_ptr<EffectEditor> PopulateOrExchange(
      ShuttleGui& S, EffectInstance& instance,
      EffectSettingsAccess& access, const EffectOutputs* pOutputs) override;

   void OnAdvancedCheckboxChanged(wxCommandEvent& evt);
   void OnDeviceInfoButtonClicked(wxCommandEvent& evt);
   void OnModelManagerButtonClicked(wxCommandEvent& evt);

protected:

   wxChoice* mTypeChoiceDeviceCtrl;
   int m_deviceSelectionChoice = 0;

   wxChoice* mTypeChoiceModelCtrl;
   int m_modelSelectionChoice = 0;

   bool TransferDataToWindow(const EffectSettings& settings) override;
   bool TransferDataFromWindow(EffectSettings& settings) override;

private:

   enum control
   {
      ID_Type = 10000,
      ID_Type_Model,
      ID_Attn_Limit,
      ID_Type_AdvancedCheckbox,
      ID_Type_DeviceInfoButton,
      ID_Type_ModelManagerButton
   };

   std::vector< std::string > mSupportedDevices;
   std::vector< EnumValueSymbol > mGuiDeviceSelections;

   std::vector< std::string > mSupportedModels;
   std::vector< EnumValueSymbol > mGuiModelSelections;

   wxCheckBox* mShowAdvancedOptionsCheckbox;
   void show_or_hide_advanced_options();
   void ClampSelectionChoices();
   bool mbAdvanced = false;

   // For little noise reduction, set to 6-12.
   // For medium, 18-24.
   // 100 means no attenuation limit
   float mAttenuationLimit = 100.0f;

   wxCheckBox* mDF3RunPostFilter;
   bool mbRunDF3PostFilter = false;

   wxSizer* attentuationLimitSizer = nullptr;
   wxSizer* noAdvancedSettingsLabel = nullptr;
   wxSizer* df3postfiltersizer = nullptr;

   wxWeakRef<wxWindow> mUIParent{};

   std::vector<std::pair<std::string, std::string>> m_simple_to_full_device_map;

   DECLARE_EVENT_TABLE()
};
