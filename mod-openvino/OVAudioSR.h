// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "StatefulEffect.h"
#include "effects/StatefulEffectUIServices.h"
#include <wx/weakref.h>
#include <mutex>

class WaveTrack;
class wxChoice;
class wxCheckBox;
class wxTextCtrl;
class wxSizer;

class AudioSR;
class EffectOVAudioSR final : public StatefulEffect, public StatefulEffectUIServices
{
public:

   static const ComponentInterfaceSymbol Symbol;

   EffectOVAudioSR();
   virtual ~EffectOVAudioSR();

   // ComponentInterface implementation
   ComponentInterfaceSymbol GetSymbol() const override;
   TranslatableString GetDescription() const override;
   VendorSymbol GetVendor() const override;

   unsigned GetAudioInCount() const override;

   // EffectDefinitionInterface implementation

   EffectType GetType() const override;
   bool IsInteractive() const override;

   bool Process(EffectInstance& instance, EffectSettings& settings) override;

   bool StepComplete(int ith_step);

   std::unique_ptr<EffectEditor> PopulateOrExchange(
      ShuttleGui& S, EffectInstance& instance,
      EffectSettingsAccess& access, const EffectOutputs* pOutputs) override;
   bool TransferDataToWindow(const EffectSettings& settings) override;
   bool TransferDataFromWindow(EffectSettings& settings) override;

   void OnAdvancedCheckboxChanged(wxCommandEvent& evt);
   void OnDeviceInfoButtonClicked(wxCommandEvent& evt);


private:

   enum control
   {
      ID_Type_ModelType = 10000,
      ID_Type_DDPMDevice,
      ID_Type_EncoderDevice,
      ID_Type_DecoderDevice,
      ID_Type_AdvancedCheckbox,
      ID_Type_DeviceInfoButton,
      ID_Type_Seed,
      ID_Type_NormalizeRMSCheckbox
   };

   std::vector< std::string > mSupportedDevices;
   std::vector< EnumValueSymbol > mGuiDeviceSelections;

   wxChoice* mTypeChoiceModelCtrl;
   int m_modelSelectionChoice = 0;
   std::vector< EnumValueSymbol > mGuiModelSelections;

   wxChoice* mTypeChoiceDeviceCtrl_Encoder;
   int m_encoderDeviceSelectionChoice = 0;
   wxChoice* mTypeChoiceDeviceCtrl_DDPM;
   int m_ddpmDeviceSelectionChoice = 0;
   wxChoice* mTypeChoiceDeviceCtrl_Decoder;
   int m_decoderDeviceSelectionChoice = 0;

   wxWeakRef<wxWindow> mUIParent{};

   wxCheckBox* mShowAdvancedOptionsCheckbox;

   wxCheckBox* mNormalizeOutputRMSCheckBox;
   bool mbNormalizeOutputRMS = true;

   int mNumSteps = 50;
   wxTextCtrl* mNumberOfStepsCtrl = nullptr;

   float mGuidanceScale = 3.5f;

   void show_or_hide_advanced_options();
   wxSizer* advancedSizer = nullptr;
   bool mbAdvanced = false;

   std::vector<std::pair<std::string, std::string>> m_simple_to_full_device_map;

   std::shared_ptr< AudioSR > _audioSR;

   int _ddim_steps_complete = 0;
   int _total_ddim_steps = 0;

   wxTextCtrl* mSeed;
   std::string _seed_str = "";

   std::mutex mProgMutex;
   float mProgressFrac = 0.f;
   std::string mProgMessage;
   bool mIsCancelled = false;

   DECLARE_EVENT_TABLE()
};

