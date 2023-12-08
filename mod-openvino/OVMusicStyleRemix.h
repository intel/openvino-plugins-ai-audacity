// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "effects/StatefulEffect.h"
#include "ShuttleAutomation.h"
#include <wx/weakref.h>

class wxString;
class wxChoice;
class wxCheckBox;
class LabelTrack;
class wxTextCtrl;
class wxButton;

class EffectOVMusicStyleRemix final : public StatefulEffect
{
public:
   static inline EffectOVMusicStyleRemix*
      FetchParameters(EffectOVMusicStyleRemix& e, EffectSettings&) { return &e; }
   static const ComponentInterfaceSymbol Symbol;

   EffectOVMusicStyleRemix();
   virtual ~EffectOVMusicStyleRemix();

   // ComponentInterface implementation

   ComponentInterfaceSymbol GetSymbol() const override;
   TranslatableString GetDescription() const override;
   VendorSymbol GetVendor() const override;

   EffectType GetType() const override;

   bool Process(EffectInstance& instance, EffectSettings& settings) override;
   std::unique_ptr<EffectEditor> PopulateOrExchange(
      ShuttleGui& S, EffectInstance& instance,
      EffectSettingsAccess& access, const EffectOutputs* pOutputs) override;
   void DoPopulateOrExchange(
      ShuttleGui& S, EffectSettingsAccess& access);
   bool TransferDataToWindow(const EffectSettings& settings) override;
   bool TransferDataFromWindow(EffectSettings& settings) override;

   bool UpdateProgress(double perc);

   bool segment_callback(size_t num_segments_complete, size_t total_segments);
   bool unet_callback(size_t unet_step_i_complete, size_t unet_total_iterations);

private:

   wxWeakRef<wxWindow> mUIParent{};

   enum control
   {
      ID_Type_TxtEncoder = 10000,
      ID_Type_VAEEncoder,
      ID_Type_UNetPositive,
      ID_Type_UNetNegative,
      ID_Type_VAEDecoder,

      ID_Type_Scheduler,

      ID_Type_StartPrompt,
      ID_Type_StartDenoising,
      ID_Type_Seed,
      ID_Type_NegativePrompt,

      ID_Type_UnloadModelsButton,
   };

   void OnUnloadModelsButtonClicked(wxCommandEvent& evt);

   wxChoice* mTypeChoiceDeviceCtrl_TextEncoder;
   wxChoice* mTypeChoiceDeviceCtrl_VAEEncoder;
   wxChoice* mTypeChoiceDeviceCtrl_UNetPositive;
   wxChoice* mTypeChoiceDeviceCtrl_UNetNegative;
   wxChoice* mTypeChoiceDeviceCtrl_VAEDecoder;


   int m_deviceSelectionChoice_TextEncoder = 0;
   int m_deviceSelectionChoice_VAEEncoder = 0;
   int m_deviceSelectionChoice_UNetPositive = 0;
   int m_deviceSelectionChoice_UNetNegative = 0;
   int m_deviceSelectionChoice_VAEDecoder = 0;

   std::vector< EnumValueSymbol > mGuiDeviceVPUSupportedSelections;
   std::vector< EnumValueSymbol > mGuiDeviceNonVPUSupportedSelections;

   EffectSettingsAccessPtr mpAccess;

   wxTextCtrl* mTextPrompt;
   float mDenoisingStrength = 0.40f;
   wxTextCtrl* mSeed;

   float mGuidanceScale = 7.f;

   int mNumInferenceSteps = 25;

   wxTextCtrl* mNegativePrompt;

   wxChoice* mTypeChoiceScheduler;
   std::vector< EnumValueSymbol > mGuiSchedulerSelections;
   int m_schedulerSelectionChoice = 0;

   size_t mSegments_complete = 0;
   size_t mNumSegments = 0;

   std::string _pos_prompt = "";
   std::string _neg_prompt = "";

   std::string _seed_str = "";

   std::mutex mProgMutex;
   float mProgressFrac = 0.f;
   std::string mProgMessage;

   wxButton* mUnloadModelsButton;

   bool mIsCancelled = false;

   DECLARE_EVENT_TABLE()
};
