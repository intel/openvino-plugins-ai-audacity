// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "OVTextToSpeechGenAI.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <wx/choice.h>
#include <wx/dir.h>
#include <wx/filename.h>
#include <wx/intl.h>
#include <wx/log.h>
#include <wx/sizer.h>
#include <wx/textctrl.h>
#include <wx/valgen.h>

#include "CodeConversions.h"
#include "EffectOutputTracks.h"
#include "LoadEffects.h"
#include "OpenVINOPluginPrefs.h"
#include "Project.h"
#include "ShuttleGui.h"
#include "LabelTrack.h"
#include "WaveTrack.h"
#include "effects/EffectEditor.h"
#include "OVModelManager.h"
#include "OVModelManagerUI.h"

#include <openvino/openvino.hpp>
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

const ComponentInterfaceSymbol EffectOVTextToSpeechGenAI::Symbol
{ XO("OpenVINO Text-to-Speech") };

namespace { BuiltinEffectsModule::Registration<EffectOVTextToSpeechGenAI> reg; }

namespace {
std::string FormatVoiceDisplayLabel(const std::string& voiceName)
{
   // Kokoro voice convention: <language><gender>_<name>
   // Examples: af_heart, bm_george, pf_dora
   if (voiceName.size() < 4 || voiceName[2] != '_') {
      return voiceName;
   }

   const char languageCode = voiceName[0];
   const char genderCode = voiceName[1];

   std::string language;
   switch (languageCode) {
   case 'a': language = "English (US)"; break;
   case 'b': language = "English (UK)"; break;
   case 'e': language = "Spanish"; break;
   case 'f': language = "French"; break;
   case 'h': language = "Hindi"; break;
   case 'i': language = "Italian"; break;
   case 'j': language = "Japanese"; break;
   case 'p': language = "Portuguese (Brazil)"; break;
   case 'z': language = "Chinese"; break;
   default:
      return voiceName;
   }

   std::string gender;
   switch (genderCode) {
   case 'm': gender = "Male"; break;
   case 'f': gender = "Female"; break;
   default:
      return voiceName;
   }

   return voiceName + " [" + language + ", " + gender + "]";
}

bool TryGetVoiceLanguagePrefix(const std::string& voiceName, char& prefix)
{
   if (voiceName.size() < 4 || voiceName[2] != '_') {
      return false;
   }

   const char candidate = voiceName[0];
   switch (candidate) {
   case 'a':
   case 'b':
   case 'e':
   case 'f':
   case 'h':
   case 'i':
   case 'j':
   case 'p':
   case 'z':
      prefix = candidate;
      return true;
   default:
      return false;
   }
}

char LanguageCodeToVoicePrefix(const std::string& languageCode)
{
   if (languageCode == "en-us") {
      return 'a';
   }
   if (languageCode == "en-gb") {
      return 'b';
   }
   if (languageCode == "es") {
      return 'e';
   }
   if (languageCode == "fr-fr") {
      return 'f';
   }
   if (languageCode == "hi") {
      return 'h';
   }
   if (languageCode == "it") {
      return 'i';
   }
   if (languageCode == "pt-br") {
      return 'p';
   }
   return '\0';
}

std::vector<std::string> ScanVoicesInModelPath(const std::string& modelPath)
{
   const wxString voicesPath = wxFileName(wxString::FromUTF8(modelPath), wxT("voices")).GetFullPath();
   if (!wxDirExists(voicesPath)) {
      return {};
   }

   wxDir voicesDir(voicesPath);
   wxString fileName;
   std::vector<std::string> voices;
   bool hasFile = voicesDir.GetFirst(&fileName, wxT("*.bin"), wxDIR_FILES);
   while (hasFile) {
      const wxFileName voiceFile(fileName);
      voices.push_back(audacity::ToUTF8(voiceFile.GetName()));
      hasFile = voicesDir.GetNext(&fileName);
   }

   std::sort(voices.begin(), voices.end());
   return voices;
}

wxString FindSpeakerEmbeddingPath(const std::string& modelPath, const std::string& selectedVoice)
{
   const wxString voicesPath = wxFileName(wxString::FromUTF8(modelPath), wxT("voices")).GetFullPath();
   if (!wxDirExists(voicesPath)) {
      throw std::runtime_error("The selected model folder does not contain a 'voices' directory.");
   }

   if (!selectedVoice.empty()) {
      const wxString requestedVoice = wxFileName(voicesPath, wxString::FromUTF8(selectedVoice + ".bin")).GetFullPath();
      if (wxFileExists(requestedVoice)) {
         return requestedVoice;
      }
   }

   const wxString preferredVoice = wxFileName(voicesPath, wxT("af_heart.bin")).GetFullPath();
   if (wxFileExists(preferredVoice)) {
      return preferredVoice;
   }

   wxDir voicesDir(voicesPath);
   wxString fileName;
   if (!voicesDir.GetFirst(&fileName, wxT("*.bin"), wxDIR_FILES)) {
      throw std::runtime_error("No speaker embedding .bin files were found in the model 'voices' directory.");
   }

   return wxFileName(voicesPath, fileName).GetFullPath();
}

size_t ShapeElementCount(const ov::Shape& shape)
{
   return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

ov::Tensor LoadSpeakerEmbeddingTensor(const wxString& speakerEmbeddingPath, const ov::Shape& expectedShape)
{
   const size_t expectedElements = ShapeElementCount(expectedShape);
   const size_t expectedBytes = expectedElements * sizeof(float);

   std::ifstream input(audacity::ToUTF8(speakerEmbeddingPath), std::ios::binary);
   if (!input) {
      throw std::runtime_error("Failed to open speaker embedding file: " + audacity::ToUTF8(speakerEmbeddingPath));
   }

   ov::Tensor speakerEmbedding(ov::element::f32, expectedShape);
   input.read(reinterpret_cast<char*>(speakerEmbedding.data<float>()), static_cast<std::streamsize>(expectedBytes));
   if (!input || static_cast<size_t>(input.gcount()) != expectedBytes) {
      throw std::runtime_error("Speaker embedding file size does not match the expected model embedding shape.");
   }

   return speakerEmbedding;
}
} // namespace

BEGIN_EVENT_TABLE(EffectOVTextToSpeechGenAI, wxEvtHandler)
   EVT_BUTTON(ID_Type_ModelManager, EffectOVTextToSpeechGenAI::OnModelManagerButtonClicked)
   EVT_CHOICE(ID_Type_TextSource, EffectOVTextToSpeechGenAI::OnTextSourceChanged)
   EVT_CHOICE(ID_Type_TtsModel, EffectOVTextToSpeechGenAI::OnTtsModelChanged)
   EVT_CHOICE(ID_Type_Language, EffectOVTextToSpeechGenAI::OnLanguageChanged)
   EVT_CHECKBOX(ID_Type_FilterVoicesByLanguage, EffectOVTextToSpeechGenAI::OnFilterVoicesByLanguageChanged)
END_EVENT_TABLE()

EffectOVTextToSpeechGenAI::EffectOVTextToSpeechGenAI()
{
   ov::Core core;
   auto devices = core.get_available_devices();

   for (const auto& d : devices) {
      if (d.find("GNA") != std::string::npos) {
         continue;
      }
      mSupportedDevices.push_back(d);
   }

   for (const auto& d : mSupportedDevices) {
      mGuiDeviceSelections.push_back({ TranslatableString{ wxString(d), {} } });
   }

   mSupportedLanguages = {
      "English (US)",
      "English (UK)",
      "Spanish",
      "French",
      "Hindi",
      "Italian",
      "Portuguese (Brazil)"
   };
   mSupportedLanguageCodes = {
      "en-us",
      "en-gb",
      "es",
      "fr-fr",
      "hi",
      "it",
      "pt-br"
   };
   for (const auto& languageLabel : mSupportedLanguages) {
      mGuiLanguageSelections.push_back({ TranslatableString{ wxString(languageLabel), {} } });
   }
}

EffectOVTextToSpeechGenAI::~EffectOVTextToSpeechGenAI() = default;

ComponentInterfaceSymbol EffectOVTextToSpeechGenAI::GetSymbol() const
{
   return Symbol;
}

bool EffectOVTextToSpeechGenAI::HasSelectedLabelTracks() const
{
   if (!mTracks) {
      return false;
   }

   for (const auto labelTrack : mTracks->Selected<LabelTrack>()) {
      if (labelTrack != nullptr) {
         return true;
      }
   }

   return false;
}

TranslatableString EffectOVTextToSpeechGenAI::GetDescription() const
{
   return XO("Generates speech from text using OpenVINO GenAI speech generation models");
}

VendorSymbol EffectOVTextToSpeechGenAI::GetVendor() const
{
   return XO("OpenVINO AI");
}

unsigned EffectOVTextToSpeechGenAI::GetAudioInCount() const
{
   return 0;
}

unsigned EffectOVTextToSpeechGenAI::GetAudioOutCount() const
{
   return 1;
}

EffectType EffectOVTextToSpeechGenAI::GetType() const
{
   return EffectTypeGenerate;
}

bool EffectOVTextToSpeechGenAI::IsInteractive() const
{
   return true;
}

std::vector<EffectOVTextToSpeechGenAI::LabelTextBlock> EffectOVTextToSpeechGenAI::ResolveSelectedLabelTrackBlocks() const
{
   constexpr double contiguousToleranceSeconds = 1e-6;
   const bool hasTimeSelection = mT1 > mT0;

   std::vector<LabelTextBlock> labels;

   for (const auto labelTrack : mTracks->Selected<LabelTrack>()) {
      for (const auto& label : labelTrack->GetLabels()) {
         if (label.title.empty()) {
            continue;
         }

         const double labelStart = label.selectedRegion.t0();
         const double labelEnd = label.selectedRegion.t1();

         bool includeLabel = true;
         if (hasTimeSelection) {
            // Only include labels with non-zero overlap; labels that only
            // touch selection boundaries should be excluded.
            includeLabel = (labelEnd - mT0) > contiguousToleranceSeconds
               && (mT1 - labelStart) > contiguousToleranceSeconds;
         }

         if (!includeLabel) {
            continue;
         }

         labels.push_back({ labelStart, labelEnd, audacity::ToUTF8(label.title) });
      }
   }

   std::sort(labels.begin(), labels.end(),
      [](const LabelTextBlock& lhs, const LabelTextBlock& rhs) {
         if (lhs.startTime != rhs.startTime) {
            return lhs.startTime < rhs.startTime;
         }
         return lhs.endTime < rhs.endTime;
      });

   std::vector<LabelTextBlock> mergedBlocks;
   for (const auto& label : labels) {
      if (mergedBlocks.empty()) {
         mergedBlocks.push_back(label);
         continue;
      }

      auto& previous = mergedBlocks.back();
      const bool isContiguous = std::abs(previous.endTime - label.startTime) <= contiguousToleranceSeconds;
      if (!isContiguous) {
         mergedBlocks.push_back(label);
         continue;
      }

      previous.endTime = std::max(previous.endTime, label.endTime);
      previous.text += label.text;
   }

   return mergedBlocks;
}

std::string EffectOVTextToSpeechGenAI::ResolvePromptText() const
{
   return mInputText;
}

bool EffectOVTextToSpeechGenAI::ApplyGeneratedBlocksToSelectedTracks(
   const std::vector<GeneratedSpeechBlock>& generatedBlocks)
{
   if (generatedBlocks.empty()) {
      return false;
   }

   EffectOutputTracks outputs { *mTracks, GetType(), { { mT0, mT1 } }, true };

   bool appliedToAnyTrack = false;

   for (auto pOutWaveTrack : outputs.Get().Selected<WaveTrack>()) {
      appliedToAnyTrack = ApplyGeneratedBlocksToTrack(*pOutWaveTrack, generatedBlocks) || appliedToAnyTrack;
   }

   if (appliedToAnyTrack) {
      outputs.Commit();
   }

   return appliedToAnyTrack;
}

bool EffectOVTextToSpeechGenAI::ApplyGeneratedBlocksToTrack(
   WaveTrack& destinationTrack,
   const std::vector<GeneratedSpeechBlock>& generatedBlocks)
{
   double latestPlacedEnd = std::numeric_limits<double>::lowest();
   bool appliedToTrack = false;

   for (const auto& block : generatedBlocks) {
      if (block.speech.empty() || block.sampleRate == 0) {
         continue;
      }

      double placementStart = block.preferredStartTime;
      if (placementStart < latestPlacedEnd) {
         placementStart = latestPlacedEnd;
      }

      auto generatedClip = destinationTrack.EmptyCopy();
      generatedClip->SetRate(block.sampleRate);
      generatedClip->Append(
         0,
         reinterpret_cast<constSamplePtr>(block.speech.data()),
         floatSample,
         block.speech.size(),
         1,
         widestSampleFormat);
      generatedClip->Flush();

      constexpr auto preserve = true;
      constexpr auto merge = true;
      destinationTrack.ClearAndPaste(
         placementStart,
         placementStart,
         *generatedClip,
         preserve,
         merge,
         nullptr);

      const double blockDuration = static_cast<double>(block.speech.size()) / static_cast<double>(block.sampleRate);
      latestPlacedEnd = placementStart + blockDuration;
      appliedToTrack = true;
   }

   return appliedToTrack;
}

bool EffectOVTextToSpeechGenAI::ApplyGeneratedBlocksToNewTrack(
   const std::vector<GeneratedSpeechBlock>& generatedBlocks)
{
   if (generatedBlocks.empty() || !mTracks || !mFactory) {
      return false;
   }

   EffectOutputTracks outputs { *mTracks, EffectTypeNone, std::nullopt, false };

   auto newOutputTrack = mFactory->Create(floatSample, mProjectRate);
   newOutputTrack->SetName(mTracks->MakeUniqueTrackName(WaveTrack::GetDefaultAudioTrackNamePreference()));
   newOutputTrack->SetSelected(true);

   bool appliedToTrack = false;
   double basePlacementStart = 0.0;
   double latestPlacedEndAbsolute = std::numeric_limits<double>::lowest();
   double latestPlacedEndRelative = 0.0;

   for (const auto& block : generatedBlocks) {
      if (block.speech.empty() || block.sampleRate == 0) {
         continue;
      }

      double placementStart = block.preferredStartTime;
      if (placementStart < latestPlacedEndAbsolute) {
         placementStart = latestPlacedEndAbsolute;
      }

      if (!appliedToTrack) {
         newOutputTrack->SetRate(block.sampleRate);
         basePlacementStart = placementStart;
      }
      else {
         const double relativeStart = placementStart - basePlacementStart;
         const double silenceDuration = relativeStart - latestPlacedEndRelative;
         if (silenceDuration > 0.0) {
            newOutputTrack->InsertSilence(latestPlacedEndRelative, silenceDuration);
         }
      }

      newOutputTrack->Append(
         0,
         reinterpret_cast<constSamplePtr>(block.speech.data()),
         floatSample,
         block.speech.size(),
         1,
         widestSampleFormat);
      // Materialize the appended block before the next timeline edit,
      // otherwise later InsertSilence operations can act on stale clip state.
      newOutputTrack->Flush();

      const double blockDuration = static_cast<double>(block.speech.size()) / static_cast<double>(block.sampleRate);
      latestPlacedEndAbsolute = placementStart + blockDuration;
      latestPlacedEndRelative = latestPlacedEndAbsolute - basePlacementStart;
      appliedToTrack = true;
   }

   if (!appliedToTrack) {
      return false;
   }

   newOutputTrack->MoveTo(basePlacementStart);

   outputs.AddToOutputTracks(newOutputTrack);
   outputs.Commit();
   return true;
}

std::string EffectOVTextToSpeechGenAI::ResolveModelPath() const
{
   const auto collection = OVModelManager::instance().GetModelCollection(OVModelManager::TtsName());
   if (!collection) {
      return {};
   }

   if (mTtsModelSelectionChoice >= 0 && mTtsModelSelectionChoice < static_cast<int>(mSupportedTtsModels.size())) {
      const std::string selectedModelName = mSupportedTtsModels[mTtsModelSelectionChoice];
      for (const auto& model_info : collection->models) {
         if (model_info->installed && model_info->model_name == selectedModelName) {
            return model_info->installation_path;
         }
      }
   }

   // Fall back: return installation path of the first installed model.
   for (const auto& model_info : collection->models) {
      if (model_info->installed) {
         return model_info->installation_path;
      }
   }

   return {};
}

void EffectOVTextToSpeechGenAI::RefreshVoicesForCurrentModel()
{
   std::string previouslySelectedVoice;
   if (mVoiceSelectionChoice >= 0 && mVoiceSelectionChoice < static_cast<int>(mVisibleVoices.size())) {
      previouslySelectedVoice = mVisibleVoices[mVoiceSelectionChoice];
   }

   mSupportedVoices.clear();
   mVisibleVoices.clear();
   mGuiVoiceSelections.clear();

   const std::string modelPath = ResolveModelPath();
   if (!modelPath.empty()) {
      mSupportedVoices = ScanVoicesInModelPath(modelPath);
   }

   char desiredPrefix = '\0';
   if (mLanguageSelectionChoice >= 0 && mLanguageSelectionChoice < static_cast<int>(mSupportedLanguageCodes.size())) {
      desiredPrefix = LanguageCodeToVoicePrefix(mSupportedLanguageCodes[mLanguageSelectionChoice]);
   }

   for (const auto& voice : mSupportedVoices) {
      char voicePrefix = '\0';
      const bool hasKnownPrefix = TryGetVoiceLanguagePrefix(voice, voicePrefix);
      if (!mFilterVoicesByLanguage || desiredPrefix == '\0' || !hasKnownPrefix || voicePrefix == desiredPrefix) {
         mVisibleVoices.push_back(voice);
      }
   }

   for (const auto& voice : mVisibleVoices) {
      mGuiVoiceSelections.push_back({ TranslatableString{ wxString(FormatVoiceDisplayLabel(voice)), {} } });
   }

   int requestedSelection = 0;
   if (!previouslySelectedVoice.empty()) {
      const auto selectedIter = std::find(mVisibleVoices.begin(), mVisibleVoices.end(), previouslySelectedVoice);
      if (selectedIter != mVisibleVoices.end()) {
         requestedSelection = static_cast<int>(std::distance(mVisibleVoices.begin(), selectedIter));
      }
   }
   else {
      const auto preferredVoice = std::find(mVisibleVoices.begin(), mVisibleVoices.end(), "af_heart");
      if (preferredVoice != mVisibleVoices.end()) {
         requestedSelection = static_cast<int>(std::distance(mVisibleVoices.begin(), preferredVoice));
      }
   }

   if (mVisibleVoices.empty()) {
      mVoiceSelectionChoice = 0;
      if (mTypeChoiceVoiceCtrl) {
         mTypeChoiceVoiceCtrl->Clear();
      }
      return;
   }

   requestedSelection = std::clamp(requestedSelection, 0, static_cast<int>(mVisibleVoices.size()) - 1);
   mVoiceSelectionChoice = requestedSelection;

   if (mTypeChoiceVoiceCtrl) {
      mTypeChoiceVoiceCtrl->Clear();
      for (const auto& voice : mVisibleVoices) {
         mTypeChoiceVoiceCtrl->Append(wxString(FormatVoiceDisplayLabel(voice)));
      }
      mTypeChoiceVoiceCtrl->SetSelection(mVoiceSelectionChoice);
   }
}

bool EffectOVTextToSpeechGenAI::GenerateSpeech(const std::string& textToSpeak)
{
   if (mDeviceSelectionChoice < 0 || mDeviceSelectionChoice >= static_cast<int>(mSupportedDevices.size())) {
      throw std::runtime_error("Invalid OpenVINO device selection.");
   }

   const std::string resolvedModelPath = ResolveModelPath();
   if (resolvedModelPath.empty()) {
      throw std::runtime_error(
         "No installed text-to-speech model was found. Use the Model Manager to check available models.");
   }

   std::string selectedVoice;
   if (mVoiceSelectionChoice >= 0 && mVoiceSelectionChoice < static_cast<int>(mVisibleVoices.size())) {
      selectedVoice = mVisibleVoices[mVoiceSelectionChoice];
   }
   const wxString speakerEmbeddingPath = FindSpeakerEmbeddingPath(resolvedModelPath, selectedVoice);

   const std::string deviceName = mSupportedDevices[mDeviceSelectionChoice];

   ov::AnyMap properties;
   std::string selectedLanguageCode = "en-us";
   if (mLanguageSelectionChoice >= 0 && mLanguageSelectionChoice < static_cast<int>(mSupportedLanguageCodes.size())) {
      selectedLanguageCode = mSupportedLanguageCodes[mLanguageSelectionChoice];
   }
   properties["language"] = selectedLanguageCode;

   if (OpenVINOPluginSettings::ReadEnableCache() && deviceName != "CPU") {
      const auto cacheFolder = FileNames::MkDir(wxFileName(OpenVINOPluginSettings::GetOrCreateCompiledModelCacheDir()).GetFullPath());
      properties[ov::cache_dir.name()] = audacity::ToUTF8(wxFileName(cacheFolder).GetFullPath());
   }

   ov::genai::Text2SpeechPipeline pipe(resolvedModelPath, deviceName);
   ov::Tensor speakerEmbedding = LoadSpeakerEmbeddingTensor(speakerEmbeddingPath, pipe.get_speaker_embedding_shape());

   std::cout << "Generating speech for text: " << textToSpeak << std::endl;
   auto result = pipe.generate(textToSpeak, speakerEmbedding, properties);
   if (result.speeches.empty()) {
      throw std::runtime_error("Text-to-speech generation returned no audio.");
   }

   const auto& waveform = result.speeches[0];
   if (waveform.get_element_type() != ov::element::f32) {
      throw std::runtime_error("Unexpected waveform type from text-to-speech model. Expected float32 output.");
   }

   mGeneratedSpeech.assign(waveform.data<const float>(), waveform.data<const float>() + waveform.get_size());
   mGeneratedSampleRate = result.output_sample_rate;
   return !mGeneratedSpeech.empty() && mGeneratedSampleRate > 0;
}

bool EffectOVTextToSpeechGenAI::Process(EffectInstance& instance, EffectSettings& settings)
{
   try {
      mGeneratedSpeech.clear();
      mGeneratedSampleRate = 0;

      const auto source = static_cast<TextSource>(mTextSourceSelectionChoice);
      if (source == TextSource::SelectedLabelTrack) {
         const auto labelBlocks = ResolveSelectedLabelTrackBlocks();
         if (labelBlocks.empty()) {
            EffectUIServices::DoMessageBox(
               *this,
               XO("No labels with text were found in the selected label track(s) for the current selection."),
               wxICON_STOP,
               XO("Error"));
            return false;
         }

         std::vector<GeneratedSpeechBlock> generatedBlocks;
         generatedBlocks.reserve(labelBlocks.size());

         for (const auto& block : labelBlocks) {
            if (!GenerateSpeech(block.text)) {
               EffectUIServices::DoMessageBox(
                  *this,
                  XO("Text-to-Speech generation produced no audio."),
                  wxICON_STOP,
                  XO("Error"));
               return false;
            }

            generatedBlocks.push_back({
               block.startTime,
               mGeneratedSpeech,
               mGeneratedSampleRate
               });
         }

         const bool applied = mGenerateIntoNewTrack
            ? ApplyGeneratedBlocksToNewTrack(generatedBlocks)
            : ApplyGeneratedBlocksToSelectedTracks(generatedBlocks);

         if (!applied) {
            EffectUIServices::DoMessageBox(
               *this,
               XO("Text-to-Speech needs at least one selected audio track to place generated snippets, or enable Generate into new track."),
               wxICON_STOP,
               XO("Error"));
            return false;
         }

         return true;
      }

      const std::string textToSpeak = ResolvePromptText();
      if (textToSpeak.empty()) {
         EffectUIServices::DoMessageBox(
            *this,
            XO("Text-to-Speech needs input text. Enter text manually or select a label track with labels in the current selection."),
            wxICON_STOP,
            XO("Error"));
         return false;
      }

      if (!GenerateSpeech(textToSpeak)) {
         EffectUIServices::DoMessageBox(
            *this,
            XO("Text-to-Speech generation produced no audio."),
            wxICON_STOP,
            XO("Error"));
         return false;
      }

      const double durationSeconds = static_cast<double>(mGeneratedSpeech.size()) / static_cast<double>(mGeneratedSampleRate);
      settings.extra.SetDuration(durationSeconds);

      return Generator::Process(instance, settings);
   }
   catch (const std::exception& error) {
      wxLogError("In Text-to-Speech effect, exception: %s", error.what());
      EffectUIServices::DoMessageBox(*this,
         XO("Text-to-Speech failed. See details in Help->Diagnostics->Show Log..."),
         wxICON_STOP,
         XO("Error"));
      return false;
   }
}

bool EffectOVTextToSpeechGenAI::GenerateTrack(const EffectSettings&, WaveTrack& tmp)
{
   if (mGeneratedSpeech.empty()) {
      return false;
   }

   // Set the track to the native sample rate of the generated audio
   tmp.SetRate(mGeneratedSampleRate);

   tmp.Append(0,
      reinterpret_cast<constSamplePtr>(mGeneratedSpeech.data()),
      floatSample,
      mGeneratedSpeech.size(),
      1,
      widestSampleFormat);

   return true;
}

std::unique_ptr<EffectEditor> EffectOVTextToSpeechGenAI::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess&, const EffectOutputs*)
{
   // Controls are re-created on each dialog open; reset cached pointers first
   // so stale pointers from a previous dialog instance are never reused.
   mTypeChoiceDeviceCtrl = nullptr;
   mTypeChoiceTextSourceCtrl = nullptr;
   mTypeChoiceTtsModelCtrl = nullptr;
   mTypeChoiceVoiceCtrl = nullptr;
   mTypeChoiceLanguageCtrl = nullptr;
   mGenerateIntoNewTrackCtrl = nullptr;
   mInputTextCtrl = nullptr;

   mSupportedTextSources.clear();
   mGuiTextSourceSelections.clear();

   const bool hasSelectedLabelTracks = HasSelectedLabelTracks();
   mSupportedTextSources.push_back("Manual text");
   if (hasSelectedLabelTracks) {
      mSupportedTextSources.push_back("Selected label track");
      mTextSourceSelectionChoice = static_cast<int>(TextSource::SelectedLabelTrack);
   }
   else {
      mTextSourceSelectionChoice = static_cast<int>(TextSource::ManualText);
   }

   for (const auto& source : mSupportedTextSources) {
      mGuiTextSourceSelections.push_back({ TranslatableString{ wxString(source), {} } });
   }

   mUIParent = S.GetParent();

   // Populate installed TTS model choices from the model manager.
   const auto collection = OVModelManager::instance().GetModelCollection(OVModelManager::TtsName());
   if (collection) {
      for (const auto& model_info : collection->models) {
         if (model_info->installed) {
            if (std::find(mSupportedTtsModels.begin(), mSupportedTtsModels.end(), model_info->model_name)
                  == mSupportedTtsModels.end()) {
               mSupportedTtsModels.push_back(model_info->model_name);
            }
         }
      }
   }

   mGuiTtsModelSelections.clear();
   for (const auto& m : mSupportedTtsModels) {
      mGuiTtsModelSelections.push_back({ TranslatableString{ wxString(m), {} } });
   }

   // Register callback so newly-installed models appear in the choice list live.
   OVModelManager::InstalledCallback callback =
      [this](const std::string& model_name) {
         wxTheApp->CallAfter([=]() {
            if (std::find(mSupportedTtsModels.begin(), mSupportedTtsModels.end(), model_name)
                  == mSupportedTtsModels.end()) {
               mSupportedTtsModels.push_back(model_name);
               mGuiTtsModelSelections.push_back({ TranslatableString{ wxString(model_name), {} } });
            }
            if (mUIParent) {
               EffectEditor::EnableApply(mUIParent, true);
               if (mTypeChoiceTtsModelCtrl) {
                  mTypeChoiceTtsModelCtrl->Append(wxString(model_name));
                  if (mTypeChoiceTtsModelCtrl->GetCount() == 1) {
                     mTypeChoiceTtsModelCtrl->SetSelection(0);
                     mTtsModelSelectionChoice = 0;
                  }
               }
               RefreshVoicesForCurrentModel();
            }
         });
      };
   OVModelManager::instance().register_installed_callback(OVModelManager::TtsName(), callback);

   if (mTtsModelSelectionChoice < 0 && !mSupportedTtsModels.empty()) {
      mTtsModelSelectionChoice = 0;
   }
   RefreshVoicesForCurrentModel();

   S.AddSpace(0, 5);
   S.StartVerticalLay();
   {
      S.StartMultiColumn(1, wxLEFT);
      {
         S.Id(ID_Type_ModelManager).AddButton(XO("Open Model Manager"));
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxEXPAND);
      {
         mTypeChoiceDeviceCtrl = S.Id(ID_Type_Device)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&mDeviceSelectionChoice)
            .AddChoice(XXO("OpenVINO Inference Device:"),
               Msgids(mGuiDeviceSelections.data(), mGuiDeviceSelections.size()));
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxEXPAND);
      {
         mTypeChoiceTtsModelCtrl = S.Id(ID_Type_TtsModel)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&mTtsModelSelectionChoice)
            .AddChoice(XXO("TTS Model:"),
               Msgids(mGuiTtsModelSelections.data(), mGuiTtsModelSelections.size()));

         mTypeChoiceVoiceCtrl = S.Id(ID_Type_Voice)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&mVoiceSelectionChoice)
            .AddChoice(XXO("Voice:"),
               Msgids(mGuiVoiceSelections.data(), mGuiVoiceSelections.size()));
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxEXPAND);
      {
         mTypeChoiceLanguageCtrl = S.Id(ID_Type_Language)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&mLanguageSelectionChoice)
            .AddChoice(XXO("Language:"),
               Msgids(mGuiLanguageSelections.data(), mGuiLanguageSelections.size()));

         S.Id(ID_Type_FilterVoicesByLanguage)
            .AddCheckBox(XXO("Filter voices by selected language"), mFilterVoicesByLanguage);
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxEXPAND);
      {
         mTypeChoiceTextSourceCtrl = S.Id(ID_Type_TextSource)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&mTextSourceSelectionChoice)
            .AddChoice(XXO("Text Source:"),
               Msgids(mGuiTextSourceSelections.data(), mGuiTextSourceSelections.size()));

         mGenerateIntoNewTrackCtrl = S.Id(ID_Type_GenerateIntoNewTrack)
            .Validator<wxGenericValidator>(&mGenerateIntoNewTrack)
            .AddCheckBox(XXO("Generate into new track"), mGenerateIntoNewTrack);
      }
      S.EndMultiColumn();

      S.AddVariableText(XO("Text:"));
      mInputTextCtrl = S.Name(XO("Text"))
         .Style(wxTE_MULTILINE)
         .MinSize(wxSize(500, 120))
         .AddTextWindow(wxString::FromUTF8(mInputText));
   }
   S.EndVerticalLay();

   return nullptr;
}

bool EffectOVTextToSpeechGenAI::TransferDataToWindow(const EffectSettings&)
{
   if (!mUIParent || !mUIParent->TransferDataToWindow()) {
      return false;
   }

   UpdateTextSourceDependentControlStates();

   const bool canApply = !mSupportedDevices.empty() && !mSupportedTtsModels.empty() && !mVisibleVoices.empty();
   if (!canApply) {
      if (mSupportedDevices.empty()) {
         wxLogInfo("OpenVINO Text-to-Speech has no supported inference devices.");
      }
      if (mSupportedTtsModels.empty()) {
         wxLogInfo("OpenVINO Text-to-Speech has no installed models. Use the Model Manager.");
      }
      if (mSupportedTtsModels.empty() == false && mVisibleVoices.empty()) {
         wxLogInfo("OpenVINO Text-to-Speech could not find voice embeddings for the selected model.");
      }
      EffectEditor::EnableApply(mUIParent, false);
   }

   return true;
}

bool EffectOVTextToSpeechGenAI::TransferDataFromWindow(EffectSettings&)
{
   if (!mUIParent || !mUIParent->Validate() || !mUIParent->TransferDataFromWindow()) {
      return false;
   }

   mInputText = audacity::ToUTF8(mInputTextCtrl->GetValue());
   return true;
}

void EffectOVTextToSpeechGenAI::UpdateInputTextEnabledState()
{
   if (!mInputTextCtrl) {
      return;
   }

   const bool manualTextSelected =
      static_cast<TextSource>(mTextSourceSelectionChoice) == TextSource::ManualText;
   mInputTextCtrl->Enable(manualTextSelected);
}

void EffectOVTextToSpeechGenAI::UpdateTextSourceDependentControlStates()
{
   UpdateInputTextEnabledState();

   if (!mGenerateIntoNewTrackCtrl) {
      return;
   }

   const bool labelTrackSelected =
      static_cast<TextSource>(mTextSourceSelectionChoice) == TextSource::SelectedLabelTrack;
   mGenerateIntoNewTrackCtrl->Enable(labelTrackSelected);
}

void EffectOVTextToSpeechGenAI::OnModelManagerButtonClicked(wxCommandEvent&)
{
   ShowModelManagerDialog();
}

void EffectOVTextToSpeechGenAI::OnTextSourceChanged(wxCommandEvent& evt)
{
   mTextSourceSelectionChoice = evt.GetSelection();
   UpdateTextSourceDependentControlStates();
}

void EffectOVTextToSpeechGenAI::OnTtsModelChanged(wxCommandEvent& evt)
{
   mTtsModelSelectionChoice = evt.GetSelection();
   RefreshVoicesForCurrentModel();
}

void EffectOVTextToSpeechGenAI::OnLanguageChanged(wxCommandEvent& evt)
{
   mLanguageSelectionChoice = evt.GetSelection();
   if (mFilterVoicesByLanguage) {
      RefreshVoicesForCurrentModel();
   }
}

void EffectOVTextToSpeechGenAI::OnFilterVoicesByLanguageChanged(wxCommandEvent& evt)
{
   mFilterVoicesByLanguage = evt.IsChecked();
   RefreshVoicesForCurrentModel();
}
