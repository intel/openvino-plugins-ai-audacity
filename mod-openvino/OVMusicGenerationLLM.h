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

#include "StatefulEffect.h"
#include "effects/StatefulEffectUIServices.h"
#include "ShuttleAutomation.h"
#include <wx/event.h>
#include <wx/choice.h>
#include <wx/weakref.h>
#include "musicgen/musicgen.h"

class EffectOVMusicGenerationLLM final : public StatefulEffect, public StatefulEffectUIServices
{
public:
   static inline EffectOVMusicGenerationLLM*
      FetchParameters(EffectOVMusicGenerationLLM& e, EffectSettings&) { return &e; }
   static const ComponentInterfaceSymbol Symbol;

   EffectOVMusicGenerationLLM();
   virtual ~EffectOVMusicGenerationLLM();

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

   virtual bool DoEffect(
      EffectSettings& settings, //!< Always given; only for processing
      const InstanceFinder& finder,
      double projectRate, TrackList* list,
      WaveTrackFactory* factory, NotifyingSelectedRegion& selectedRegion,
      unsigned flags,
      const EffectSettingsAccessPtr& pAccess = nullptr
      //!< Sometimes given; only for UI
   ) override;


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

      ID_Type_DeviceInfoButton
   };

   void OnContextLengthChanged(wxCommandEvent& evt);
   void OnUnloadModelsButtonClicked(wxCommandEvent& evt);
   void OnDeviceInfoButtonClicked(wxCommandEvent& evt);

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
   static constexpr EffectParameter RMSLevel{ &EffectOVMusicGenerationLLM::mRMSLevel,
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

   std::vector<std::pair<std::string, std::string>> m_simple_to_full_device_map;

   // The number of tracks that the user has selected upon selecting this generator from the menu.
   // If it's 0 (i.e. they had no tracks selected), then EffectBase will create a new mono track on
   // behalf of the user. In this special (but common) case, if they have selected a stereo model,
   // then we will replace the one that EffectBase added with a stereo one.
   size_t _num_selected_tracks_at_effect_start = 0;

   DECLARE_EVENT_TABLE()
};


