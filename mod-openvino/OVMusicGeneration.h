// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

class wxString;
class wxCheckBox;
class LabelTrack;
class wxTextCtrl;
class wxFlexGridSizer;
class wxWindow;
class NumericTextCtrl;
class wxButton;

#include "effects/StatefulEffect.h"
#include "ShuttleAutomation.h"
#include <wx/event.h>
#include <wx/choice.h>
#include <wx/weakref.h>

class EffectOVMusicGeneration final : public StatefulEffect
{
public:
   static inline EffectOVMusicGeneration*
      FetchParameters(EffectOVMusicGeneration& e, EffectSettings&) { return &e; }
   static const ComponentInterfaceSymbol Symbol;

   EffectOVMusicGeneration();
   virtual ~EffectOVMusicGeneration();

   // ComponentInterface implementation

   ComponentInterfaceSymbol GetSymbol() const override;
   TranslatableString GetDescription() const override;
   VendorSymbol GetVendor() const override;

   // EffectDefinitionInterface implementation

   EffectType GetType() const override;


   // Effect implementation
   bool Process(EffectInstance& instance, EffectSettings& settings) override;
   std::unique_ptr<EffectEditor> PopulateOrExchange(
      ShuttleGui& S, EffectInstance& instance,
      EffectSettingsAccess& access, const EffectOutputs* pOutputs) override;
   void DoPopulateOrExchange(
      ShuttleGui& S, EffectSettingsAccess& access);
   bool TransferDataToWindow(const EffectSettings& settings) override;
   bool TransferDataFromWindow(EffectSettings& settings) override;

   bool UpdateProgress(double perc);

   bool interp_callback(size_t interp_step_i_complete);
   bool unet_callback_for_interp(size_t unet_step_i_complete, size_t unet_total_iterations);
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

      ID_Type_Mode,

      ID_Type_SeedImageSimple,
      ID_Type_SeedImage,

      ID_Type_Scheduler,

      ID_Type_StartPrompt,
      ID_Type_StartDenoising,
      ID_Type_StartSeed,

      ID_Type_EndPrompt,
      ID_Type_EndDenoising,
      ID_Type_EndSeed,

      ID_Type_NegativePromptAdvanced,

      ID_Type_Prompt,
      ID_Type_NegativePrompt,
      ID_Type_Seed,

      ID_Type_NumInterpolationSteps,
      ID_Type_Duration,

      ID_Type_UnloadModelsButton,

   };

   void OnChoice(wxCommandEvent& evt);
   void OnSimpleSeedImageChoice(wxCommandEvent& evt);
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

   //Start of Advanced Settings
   wxSizer* advancedSizer0 = nullptr;
   wxSizer* advancedSizer1 = nullptr;
   wxTextCtrl* mTextPromptStart;
   float mStartDenoising = 0.75f;
   wxTextCtrl* mStartSeed;

   wxTextCtrl* mTextPromptEnd;
   float mEndDenoising = 0.75f;
   wxTextCtrl* mEndSeed;

   wxTextCtrl* mTextPromptNumInterpolationSteps;
   int mNumInterpolationSteps = 10;
   int mNumOutputSegments = 2;

   size_t mNumTotalUnetIterations = 1;
   size_t mNumUnetIterationsComplete = 0;

   wxChoice* mTypeChoiceSeedImageSimple;
   std::vector< EnumValueSymbol > mGuiSeedImageSelectionsSimple;
   int m_seedImageSelectionChoiceSimple = 0;

   wxChoice* mTypeChoiceSeedImage;
   std::vector< EnumValueSymbol > mGuiSeedImageSelections;
   int m_seedImageSelectionChoice = 0;

   wxTextCtrl* mNegativePromptAdvanced;

   // Start of Simple settings
   wxSizer* simpleSizer0 = nullptr;
   wxTextCtrl* mTextPrompt;
   wxTextCtrl* mSeed;
   //wxTextCtrl* mNegativePrompt;

   std::string _start_seed_str="";
   std::string _end_seed_str = "";

   float mGuidanceScaleAdvanced = 7.5f;
   float mGuidanceScale = 7.5f;

   int mNumInferenceStepsAdvanced = 20;

   int mNumInferenceSteps = 20;

   float mDenoisingSimple = 0.75f;
   wxTextCtrl* mDenoisingSimpleCtl;

   wxChoice* mTypeChoiceScheduler;
   std::vector< EnumValueSymbol > mGuiSchedulerSelections;
   int m_schedulerSelectionChoice = 0;

   size_t mInterp_steps_complete = 0;

   std::string _pos_prompt_start="";
   std::string _pos_prompt_end="";
   std::string _neg_prompt_end="";

   wxChoice* mChoiceMode;
   std::vector< EnumValueSymbol > mGuiModeSelections;
   int m_modeSelectionChoice = 0;

   wxButton* mUnloadModelsButton;

   NumericTextCtrl* mDurationT;

   std::mutex mProgMutex;
   float mProgressFrac = 0.f;
   std::string mProgMessage;

   double mRMSLevel = -20.0;
   static constexpr EffectParameter RMSLevel{ &EffectOVMusicGeneration::mRMSLevel,
   L"RMSLevel",            -20.0,      -145.0,  0.0,      1 };

   bool mIsCancelled = false;

   DECLARE_EVENT_TABLE()
};


