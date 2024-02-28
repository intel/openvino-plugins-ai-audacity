// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once
#include <mutex>
#include "effects/StatefulEffect.h"
#include "ShuttleAutomation.h"

class wxString;
class wxChoice;
class wxCheckBox;
class LabelTrack;
class wxTextCtrl;
class wxSizer;

#include <wx/weakref.h>


class EffectOVWhisperTranscription final : public StatefulEffect
{
public:
   //static inline EffectOVWhisperTranscription*
   //   FetchParameters(EffectOVWhisperTranscription& e, EffectSettings&) { return &e; }
   static const ComponentInterfaceSymbol Symbol;

   EffectOVWhisperTranscription();
   virtual ~EffectOVWhisperTranscription();

   // ComponentInterface implementation

   ComponentInterfaceSymbol GetSymbol() const override;
   TranslatableString GetDescription() const override;
   VendorSymbol GetVendor() const override;

   // EffectDefinitionInterface implementation

   EffectType GetType() const override;

   bool Process(EffectInstance& instance, EffectSettings& settings) override;
   std::unique_ptr<EffectEditor> PopulateOrExchange(
      ShuttleGui& S, EffectInstance& instance,
      EffectSettingsAccess& access, const EffectOutputs* pOutputs) override;

   bool UpdateProgress(double perc);
   bool _EncoderBegin();

   void OnAdvancedCheckboxChanged(wxCommandEvent& evt);

private:
   // EffectFindCliping implementation

   int mStart;   ///< Using int rather than sampleCount because values are only ever small numbers
   int mStop;    ///< Using int rather than sampleCount because values are only ever small numbers
   // To do: eliminate this
   EffectSettingsAccessPtr mpAccess;

   //const EffectParameterMethods& Parameters() const override;
   bool ProcessStereoToMono(sampleCount& curTime, sampleCount totalTime, WaveTrack& track);

   bool ProcessWhisper(WaveTrack* mono, LabelTrack* lt0, LabelTrack* lt1);
   bool Whisper(std::vector<float>& mono_samples, LabelTrack* lt0, LabelTrack* lt1, double start_time);

   wxWeakRef<wxWindow> mUIParent{};

   enum control
   {
      ID_Type = 10000,
      ID_Type_Model = 10001,
      ID_Type_Mode = 10002,
      ID_Type_Language = 10003,
      ID_Type_AdvancedCheckbox = 10004,
   };

   wxChoice* mTypeChoiceDeviceCtrl;
   int m_deviceSelectionChoice = 0;

   std::vector< std::string > mSupportedDevices;
   std::vector< EnumValueSymbol > mGuiDeviceSelections;

   wxChoice* mTypeChoiceModelCtrl;
   int m_modelSelectionChoice = 0;

   std::vector< std::string > mSupportedModels;
   std::vector< EnumValueSymbol > mGuiModelSelections;

   wxChoice* mTypeChoiceModeCtrl;
   int m_modeSelectionChoice = 0;

   std::vector< std::string > mSupportedModes;
   std::vector< EnumValueSymbol > mGuiModeSelections;

   wxChoice* mTypeChoiceLanguageCtrl;
   int m_languageSelectionChoice = 0;

   std::vector< std::string > mSupportedLanguages;
   std::vector< EnumValueSymbol > mGuiLanguageSelections;

   //static constexpr EffectParameter Start{ &EffectTranscription::mStart,
   //   L"Duty Cycle Start",  3,    1,    INT_MAX, 1 };
   //static constexpr EffectParameter Stop{ &EffectTranscription::mStop,
   //   L"Duty Cycle End",    3,    1,    INT_MAX, 1 };

   std::mutex mMutex;
   bool mIsCancelled = false;

   void show_or_hide_advanced_options();

   //advanced options
   wxCheckBox* mShowAdvancedOptionsCheckbox;
   bool mbAdvanced = false;

   wxSizer* advancedSizer = nullptr;

   int mMaxTextSegLength = 0;
   wxTextCtrl* mMaxTextSegLengthCtrl = nullptr;

   std::string mInitialPrompt = "";
   wxTextCtrl* mInitialPromptCtrl = nullptr;

   int mBeamSize = 1;
   wxTextCtrl* mBeamSizeCtrl = nullptr;

   int mBestOf = 1;
   wxTextCtrl* mBestOfCtrl = nullptr;

   DECLARE_EVENT_TABLE()
};
