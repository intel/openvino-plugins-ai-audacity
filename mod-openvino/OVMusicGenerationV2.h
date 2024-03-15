// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

class wxString;
class wxStaticText;
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
#include "musicgen/musicgen.h"

class EffectOVMusicGenerationV2 final : public StatefulEffect
{
public:
   static inline EffectOVMusicGenerationV2*
      FetchParameters(EffectOVMusicGenerationV2& e, EffectSettings&) { return &e; }
   static const ComponentInterfaceSymbol Symbol;

   EffectOVMusicGenerationV2();
   virtual ~EffectOVMusicGenerationV2();

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

   bool MusicGenCallback(float perc_complete);

private:

   wxWeakRef<wxWindow> mUIParent{};

   enum control
   {
      ID_Type_EnCodec = 10000,

      ID_Type_MusicGenDecodeDevice0,
      ID_Type_MusicGenDecodeDevice1,

      ID_Type_Prompt,
      ID_Type_Seed,

      ID_Type_Duration,

      ID_Type_UnloadModelsButton,

      ID_Type_ContextLength,
      ID_Type_ModelSelection,

      ID_Type_GuidanceScale,
      ID_Type_TopK,

      ID_Type_AudioContinuationCheckBox,
      ID_Type_AudioContinuationAsNewTrackCheckBox,
   };

   void OnContextLengthChanged(wxCommandEvent& evt);
   void OnUnloadModelsButtonClicked(wxCommandEvent& evt);

   wxChoice* mTypeChoiceDeviceCtrl_EnCodec;
   wxChoice* mTypeChoiceDeviceCtrl_Decode0;
   wxChoice* mTypeChoiceDeviceCtrl_Decode1;


   int m_deviceSelectionChoice_EnCodec = 0;
   int m_deviceSelectionChoice_MusicGenDecode0 = 0;
   int m_deviceSelectionChoice_MusicGenDecode1 = 0;

   std::vector< EnumValueSymbol > mGuiDeviceVPUSupportedSelections;
   std::vector< EnumValueSymbol > mGuiDeviceNonVPUSupportedSelections;

   EffectSettingsAccessPtr mpAccess;

   wxTextCtrl* mTextPrompt;

   //just default it to something right now.
   std::string _prompt = "80s pop track with bassy drums and synth";

   wxTextCtrl* mSeed;
   std::string _seed_str = "";

   wxButton* mUnloadModelsButton;

   NumericTextCtrl* mDurationT = nullptr;
   double _previous_duration;

   std::mutex mProgMutex;
   float mProgressFrac = 0.f;
   std::string mProgMessage;

   double mRMSLevel = -20.0;
   static constexpr EffectParameter RMSLevel{ &EffectOVMusicGenerationV2::mRMSLevel,
   L"RMSLevel",            -20.0,      -145.0,  0.0,      1 };

   bool mIsCancelled = false;

   std::shared_ptr<ov_musicgen::MusicGen> _musicgen;
   ov_musicgen::MusicGenConfig _musicgen_config;

   std::vector< EnumValueSymbol > mGuiContextLengthSelections;
   int m_contextLengthChoice = 1; //default to 10s
   wxChoice* mTypeChoiceContextLength;

   std::vector< std::string > mModelSelections;
   std::vector< EnumValueSymbol > mGuiModelSelections;
   int m_modelSelectionChoice = 0;
   wxChoice* mTypeChoiceModelSelection;

   wxCheckBox* _AudioContinuationCheckBox;
   wxCheckBox* _AudioContinuationAsNewTrackCheckBox;

   float mGuidanceScale = 3.0f;

   int mTopK = 50;
   wxTextCtrl* mTopKCtl;

   void SetContinuationContextWarning();
   wxStaticText* _continuationContextWarning = nullptr;

   DECLARE_EVENT_TABLE()
};


