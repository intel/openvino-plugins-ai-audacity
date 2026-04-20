// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <string>
#include <vector>

#include "Generator.h"
#include "effects/StatefulEffectUIServices.h"

#include <wx/weakref.h>

class LabelTrack;
class wxChoice;
class wxTextCtrl;

class EffectOVTextToSpeechGenAI final : public Generator, public StatefulEffectUIServices
{
public:
   static const ComponentInterfaceSymbol Symbol;

   EffectOVTextToSpeechGenAI();
   ~EffectOVTextToSpeechGenAI() override;

   ComponentInterfaceSymbol GetSymbol() const override;
   TranslatableString GetDescription() const override;
   VendorSymbol GetVendor() const override;

   unsigned GetAudioInCount() const override;
   unsigned GetAudioOutCount() const override;

   EffectType GetType() const override;
   bool IsInteractive() const override;

   bool Process(EffectInstance& instance, EffectSettings& settings) override;

   std::unique_ptr<EffectEditor> PopulateOrExchange(
      ShuttleGui& S, EffectInstance& instance,
      EffectSettingsAccess& access, const EffectOutputs* pOutputs) override;

protected:
   bool TransferDataToWindow(const EffectSettings& settings) override;
   bool TransferDataFromWindow(EffectSettings& settings) override;

private:
   struct LabelTextBlock
   {
      double startTime = 0.0;
      double endTime = 0.0;
      std::string text;
   };

   struct GeneratedSpeechBlock
   {
      double preferredStartTime = 0.0;
      std::vector<float> speech;
      uint32_t sampleRate = 0;
   };

   enum control
   {
      ID_Type_Device = 14000,
      ID_Type_TextSource,
      ID_Type_TtsModel,
      ID_Type_ModelManager,
      ID_Type_Voice,
      ID_Type_Language
   };

   enum class TextSource
   {
      ManualText = 0,
      SelectedLabelTrack = 1
   };

   bool GenerateTrack(const EffectSettings& settings, WaveTrack& tmp) override;

   std::string ResolvePromptText() const;
   std::vector<LabelTextBlock> ResolveSelectedLabelTrackBlocks() const;
   bool GenerateSpeech(const std::string& textToSpeak);
   bool ApplyGeneratedBlocksToSelectedTracks(const std::vector<GeneratedSpeechBlock>& generatedBlocks);
   std::string ResolveModelPath() const;
   void RefreshVoicesForCurrentModel();

   void OnModelManagerButtonClicked(wxCommandEvent& evt);
   void OnTtsModelChanged(wxCommandEvent& evt);

   wxWeakRef<wxWindow> mUIParent{};

   wxChoice* mTypeChoiceDeviceCtrl{};
   int mDeviceSelectionChoice = 0;
   std::vector<std::string> mSupportedDevices;
   std::vector<EnumValueSymbol> mGuiDeviceSelections;

   wxChoice* mTypeChoiceTextSourceCtrl{};
   int mTextSourceSelectionChoice = 0;
   std::vector<std::string> mSupportedTextSources;
   std::vector<EnumValueSymbol> mGuiTextSourceSelections;

   wxChoice* mTypeChoiceTtsModelCtrl{};
   int mTtsModelSelectionChoice = 0;
   std::vector<std::string> mSupportedTtsModels;
   std::vector<EnumValueSymbol> mGuiTtsModelSelections;

   wxChoice* mTypeChoiceVoiceCtrl{};
   int mVoiceSelectionChoice = 0;
   std::vector<std::string> mSupportedVoices;
   std::vector<EnumValueSymbol> mGuiVoiceSelections;

   wxChoice* mTypeChoiceLanguageCtrl{};
   int mLanguageSelectionChoice = 0;
   std::vector<std::string> mSupportedLanguages;
   std::vector<std::string> mSupportedLanguageCodes;
   std::vector<EnumValueSymbol> mGuiLanguageSelections;

   wxTextCtrl* mInputTextCtrl{};

   std::string mInputText;

   std::vector<float> mGeneratedSpeech;
   uint32_t mGeneratedSampleRate = 0;

   DECLARE_EVENT_TABLE()
};
