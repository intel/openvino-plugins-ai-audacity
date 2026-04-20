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
#include <wx/dirdlg.h>
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

#include <openvino/openvino.hpp>
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

const ComponentInterfaceSymbol EffectOVTextToSpeechGenAI::Symbol
{ XO("OpenVINO Text-to-Speech") };

namespace { BuiltinEffectsModule::Registration<EffectOVTextToSpeechGenAI> reg; }

namespace {
wxString FindSpeakerEmbeddingPath(const wxString& modelPath)
{
   const wxString voicesPath = wxFileName(modelPath, wxT("voices")).GetFullPath();
   if (!wxDirExists(voicesPath)) {
      throw std::runtime_error("The selected model folder does not contain a 'voices' directory.");
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
   EVT_BUTTON(ID_Type_BrowseModelPath, EffectOVTextToSpeechGenAI::OnBrowseModelPath)
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

   mSupportedTextSources = { "Manual text", "Selected label track" };
   for (const auto& source : mSupportedTextSources) {
      mGuiTextSourceSelections.push_back({ TranslatableString{ wxString(source), {} } });
   }
}

EffectOVTextToSpeechGenAI::~EffectOVTextToSpeechGenAI() = default;

ComponentInterfaceSymbol EffectOVTextToSpeechGenAI::GetSymbol() const
{
   return Symbol;
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
            includeLabel = labelEnd >= mT0 && labelStart <= mT1;
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
      double latestPlacedEnd = std::numeric_limits<double>::lowest();

      for (const auto& block : generatedBlocks) {
         if (block.speech.empty() || block.sampleRate == 0) {
            continue;
         }

         double placementStart = block.preferredStartTime;
         if (placementStart < latestPlacedEnd) {
            placementStart = latestPlacedEnd;
         }

         auto generatedClip = pOutWaveTrack->EmptyCopy();
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
         pOutWaveTrack->ClearAndPaste(
            placementStart,
            placementStart,
            *generatedClip,
            preserve,
            merge,
            nullptr);

         const double blockDuration = static_cast<double>(block.speech.size()) / static_cast<double>(block.sampleRate);
         latestPlacedEnd = placementStart + blockDuration;
         appliedToAnyTrack = true;
      }
   }

   if (appliedToAnyTrack) {
      outputs.Commit();
   }

   return appliedToAnyTrack;
}

wxString EffectOVTextToSpeechGenAI::ResolveModelPath() const
{
   if (!mModelPath.empty()) {
      return mModelPath;
   }

   const wxString openvinoModelsPath = OpenVINOPluginSettings::GetOrCreateModelDir(true);
   const wxString speechGenerationDir = wxFileName(openvinoModelsPath, wxT("speech_generation")).GetFullPath();

   if (!wxDirExists(speechGenerationDir)) {
      return {};
   }

   wxDir speechDir(speechGenerationDir);

   wxString subDirName;
   bool hasDir = speechDir.GetFirst(&subDirName, wxEmptyString, wxDIR_DIRS);
   while (hasDir) {
      const wxString candidate = wxFileName(speechGenerationDir, subDirName).GetFullPath();
      if (wxDirExists(wxFileName(candidate, wxT("voices")).GetFullPath())) {
         return candidate;
      }
      hasDir = speechDir.GetNext(&subDirName);
   }

   return {};
}

bool EffectOVTextToSpeechGenAI::GenerateSpeech(const std::string& textToSpeak)
{
   if (mDeviceSelectionChoice < 0 || mDeviceSelectionChoice >= static_cast<int>(mSupportedDevices.size())) {
      throw std::runtime_error("Invalid OpenVINO device selection.");
   }

   const wxString resolvedModelPath = ResolveModelPath();
   if (resolvedModelPath.empty()) {
      throw std::runtime_error(
         "No text-to-speech model folder is configured. Set Model Folder to an exported OpenVINO GenAI speech model.");
   }

   const wxString speakerEmbeddingPath = FindSpeakerEmbeddingPath(resolvedModelPath);

   const std::string deviceName = mSupportedDevices[mDeviceSelectionChoice];

   ov::AnyMap properties;
   properties["language"] = std::string("en-us");

   if (OpenVINOPluginSettings::ReadEnableCache() && deviceName != "CPU") {
      const auto cacheFolder = FileNames::MkDir(wxFileName(OpenVINOPluginSettings::GetOrCreateCompiledModelCacheDir()).GetFullPath());
      properties[ov::cache_dir.name()] = audacity::ToUTF8(wxFileName(cacheFolder).GetFullPath());
   }

   ov::genai::Text2SpeechPipeline pipe(audacity::ToUTF8(resolvedModelPath), deviceName);
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

         if (!ApplyGeneratedBlocksToSelectedTracks(generatedBlocks)) {
            EffectUIServices::DoMessageBox(
               *this,
               XO("Text-to-Speech needs at least one selected audio track to place generated snippets."),
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
   mUIParent = S.GetParent();

   if (mModelPath.empty()) {
      mModelPath = ResolveModelPath();
   }

   S.AddSpace(0, 5);
   S.StartVerticalLay();
   {
      S.StartMultiColumn(2, wxEXPAND);
      {
         mTypeChoiceDeviceCtrl = S.Id(ID_Type_Device)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&mDeviceSelectionChoice)
            .AddChoice(XXO("OpenVINO Inference Device:"),
               Msgids(mGuiDeviceSelections.data(), mGuiDeviceSelections.size()));

         mTypeChoiceTextSourceCtrl = S.Id(ID_Type_TextSource)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&mTextSourceSelectionChoice)
            .AddChoice(XXO("Text Source:"),
               Msgids(mGuiTextSourceSelections.data(), mGuiTextSourceSelections.size()));
      }
      S.EndMultiColumn();

      S.StartMultiColumn(1, wxEXPAND);
      {
         mInputTextCtrl = S.Style(wxTE_LEFT | wxTE_MULTILINE)
            .AddTextBox(XXO("Text:"), wxString::FromUTF8(mInputText), 50);
      }
      S.EndMultiColumn();

      S.StartMultiColumn(3, wxEXPAND);
      {
         mModelPathCtrl = S.Id(ID_Type_ModelPath)
            .Style(wxTE_LEFT)
            .AddTextBox(XXO("Model Folder:"), mModelPath, 40);

         S.Id(ID_Type_BrowseModelPath).AddButton(XO("Browse..."));
      }
      S.EndMultiColumn();
   }
   S.EndVerticalLay();

   return nullptr;
}

bool EffectOVTextToSpeechGenAI::TransferDataToWindow(const EffectSettings&)
{
   if (!mUIParent || !mUIParent->TransferDataToWindow()) {
      return false;
   }

   if (mSupportedDevices.empty()) {
      wxLogInfo("OpenVINO Text-to-Speech has no supported inference devices.");
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
   mModelPath = mModelPathCtrl->GetValue();
   return true;
}

void EffectOVTextToSpeechGenAI::OnBrowseModelPath(wxCommandEvent&)
{
   wxDirDialog dialog(
      mUIParent.get(),
      XO("Choose an OpenVINO GenAI text-to-speech model directory").Translation(),
      mModelPath.empty() ? OpenVINOPluginSettings::GetOrCreateModelDir(true) : mModelPath,
      wxDD_DEFAULT_STYLE | wxDD_DIR_MUST_EXIST);

   if (dialog.ShowModal() == wxID_OK && mModelPathCtrl) {
      mModelPathCtrl->SetValue(dialog.GetPath());
   }
}
