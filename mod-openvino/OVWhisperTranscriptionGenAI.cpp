// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "OVWhisperTranscriptionGenAI.h"
#include "LoadEffects.h"
#include "WaveTrack.h"
#include "EffectOutputTracks.h"
#include "effects/EffectEditor.h"
#include <future>
#include <chrono>


#include <math.h>

#include <wx/intl.h>
#include <wx/choice.h>
#include <wx/intl.h>
#include <wx/valgen.h>
#include <wx/checkbox.h>
#include <wx/log.h>
#include <wx/textctrl.h>
#include <wx/sizer.h>

#include "AnalysisTracks.h"
#include "ShuttleGui.h"
#include "widgets/valnum.h"

#include "BasicUI.h"

#include "LabelTrack.h"
#include "WaveTrack.h"
#include "whisper.h"

#include "Mix.h"
#include "MixAndRender.h"
#include "Project.h"
#include "FileNames.h"
#include "CodeConversions.h"

#include "OVStringUtils.h"

#include <openvino/openvino.hpp>
#include "openvino/genai/whisper_pipeline.hpp"

struct ovgenai_whisper_lang_entry
{
   std::string lang_full_string;
   std::string lang_code_string;
};

static const std::vector< ovgenai_whisper_lang_entry > g_lang_entries = {
    { "english", "<|en|>" },
    { "afrikaans", "<|af|>" },
    { "albanian", "<|sq|>" },
    { "amharic", "<|am|>" },
    { "arabic", "<|ar|>" },
    { "armenian", "<|hy|>" },
    { "assamese", "<|as|>" },
    { "azerbaijani", "<|az|>" },
    { "bashkir", "<|ba|>" },
    { "basque", "<|eu|>" },
    { "belarusian", "<|be|>" },
    { "bengali", "<|bn|>" },
    { "bosnian", "<|bs|>" },
    { "breton", "<|br|>" },
    { "bulgarian", "<|bg|>" },
    { "catalan", "<|ca|>" },
    { "chinese", "<|zh|>" },
    { "croatian", "<|hr|>" },
    { "czech", "<|cs|>" },
    { "danish", "<|da|>" },
    { "dutch", "<|nl|>" },
    { "estonian", "<|et|>" },
    { "faroese", "<|fo|>" },
    { "finnish", "<|fi|>" },
    { "french", "<|fr|>" },
    { "galician", "<|gl|>" },
    { "georgian", "<|ka|>" },
    { "german", "<|de|>" },
    { "greek", "<|el|>" },
    { "gujarati", "<|gu|>" },
    { "haitian creole", "<|ht|>" },
    { "hausa", "<|ha|>" },
    { "hawaiian", "<|haw|>" },
    { "hebrew", "<|he|>" },
    { "hindi", "<|hi|>" },
    { "hungarian", "<|hu|>" },
    { "icelandic", "<|is|>" },
    { "indonesian", "<|id|>" },
    { "italian", "<|it|>" },
    { "japanese", "<|ja|>" },
    { "javanese", "<|jw|>" },
    { "kannada", "<|kn|>" },
    { "kazakh", "<|kk|>" },
    { "khmer", "<|km|>" },
    { "korean", "<|ko|>" },
    { "lao", "<|lo|>" },
    { "latin", "<|la|>" },
    { "latvian", "<|lv|>" },
    { "lingala", "<|ln|>" },
    { "lithuanian", "<|lt|>" },
    { "luxembourgish", "<|lb|>" },
    { "macedonian", "<|mk|>" },
    { "malagasy", "<|mg|>" },
    { "malay", "<|ms|>" },
    { "malayalam", "<|ml|>" },
    { "maltese", "<|mt|>" },
    { "maori", "<|mi|>" },
    { "marathi", "<|mr|>" },
    { "mongolian", "<|mn|>" },
    { "myanmar", "<|my|>" },
    { "nepali", "<|ne|>" },
    { "norwegian", "<|no|>" },
    { "nynorsk", "<|nn|>" },
    { "occitan", "<|oc|>" },
    { "pashto", "<|ps|>" },
    { "persian", "<|fa|>" },
    { "polish", "<|pl|>" },
    { "portuguese", "<|pt|>" },
    { "punjabi", "<|pa|>" },
    { "romanian", "<|ro|>" },
    { "russian", "<|ru|>" },
    { "sanskrit", "<|sa|>" },
    { "serbian", "<|sr|>" },
    { "shona", "<|sn|>" },
    { "sindhi", "<|sd|>" },
    { "sinhala", "<|si|>" },
    { "slovak", "<|sk|>" },
    { "slovenian", "<|sl|>" },
    { "somali", "<|so|>" },
    { "spanish", "<|es|>" },
    { "sundanese", "<|su|>" },
    { "swahili", "<|sw|>" },
    { "swedish", "<|sv|>" },
    { "tagalog", "<|tl|>" },
    { "tajik", "<|tg|>" },
    { "tamil", "<|ta|>" },
    { "tatar", "<|tt|>" },
    { "telugu", "<|te|>" },
    { "thai", "<|th|>" },
    { "tibetan", "<|bo|>" },
    { "turkish", "<|tr|>" },
    { "turkmen", "<|tk|>" },
    { "ukrainian", "<|uk|>" },
    { "urdu", "<|ur|>" },
    { "uzbek", "<|uz|>" },
    { "vietnamese", "<|vi|>" },
    { "welsh", "<|cy|>" },
    { "yiddish", "<|yi|>" },
    { "yoruba", "<|yo|>" },
};

const ComponentInterfaceSymbol EffectOVWhisperTranscriptionGenAI::Symbol
{ XO("OpenVINO Whisper Transcription (GenAI)") };

namespace { BuiltinEffectsModule::Registration< EffectOVWhisperTranscriptionGenAI > reg; }

// hack to get around 'AddAnalysisTrack' and 'ModifyAnalysisTrack' not tagged with AUDACITY_DLL_API
static std::shared_ptr<AddedAnalysisTrack> MyAddAnalysisTrack(
   Effect& effect, const wxString& name)
{
   return std::shared_ptr<AddedAnalysisTrack>
   { safenew AddedAnalysisTrack{ &effect, name } };
}

// Set name to given value if that is not empty, else use default name
static ModifiedAnalysisTrack MyModifyAnalysisTrack(
   Effect& effect, const LabelTrack& origTrack, const wxString& name)
{
   return{ &effect, origTrack, name };
}

void EffectOVWhisperTranscriptionGenAI::_process_available_model(const std::string& ui_name,
   const std::string& folder_name,
   std::vector<std::string>& available_models)
{
   auto model_folder = wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath();
   auto base_whisper_folder = wxFileName(model_folder, wxT("whisper")).GetFullPath();

   auto whisper_model_folder = wxFileName(base_whisper_folder, wxString(folder_name)).GetFullPath();
   auto decoder = wxFileName(whisper_model_folder, wxString("openvino_decoder_model.xml"));
   auto encoder = wxFileName(whisper_model_folder, wxString("openvino_encoder_model.xml"));
   auto tokenizer = wxFileName(whisper_model_folder, wxString("openvino_tokenizer.xml"));
   if (decoder.FileExists() && encoder.FileExists() && tokenizer.FileExists())
   {
      _ui_name_to_model_info[ui_name] = { ui_name, audacity::ToUTF8(whisper_model_folder) };
      available_models.push_back(ui_name);
   }
}

std::vector<std::string> EffectOVWhisperTranscriptionGenAI::_FindAvailableModels()
{
   std::vector<std::string> available_models;

   _process_available_model("Whisper Base (FP16)", "whisper-base-fp16-ov", available_models);

   _process_available_model("Whisper Medium (INT4)", "whisper-medium-int4-ov", available_models);

   _process_available_model("Distil-Whisper V3 (INT4)", "distil-whisper-large-v3-int4-ov", available_models);

   
   return available_models;
}

BEGIN_EVENT_TABLE(EffectOVWhisperTranscriptionGenAI, wxEvtHandler)
    EVT_CHECKBOX(ID_Type_AdvancedCheckbox, EffectOVWhisperTranscriptionGenAI::OnAdvancedCheckboxChanged)
    EVT_BUTTON(ID_Type_DeviceInfoButton, EffectOVWhisperTranscriptionGenAI::OnDeviceInfoButtonClicked)
END_EVENT_TABLE()

EffectOVWhisperTranscriptionGenAI::EffectOVWhisperTranscriptionGenAI()
{
   ov::Core core;
   auto devices = core.get_available_devices();

   for (auto d : devices)
   {
      //GNA devices are not supported
      if (d.find("GNA") != std::string::npos) continue;

      m_simple_to_full_device_map.push_back({ d, core.get_property(d, "FULL_DEVICE_NAME").as<std::string>() });

      mSupportedDevices.push_back(d);
   }

   for (auto d : mSupportedDevices)
   {
      mGuiDeviceSelections.push_back({ TranslatableString{ wxString(d), {}} });
   }

   mSupportedModels = _FindAvailableModels();
   for (auto &m : mSupportedModels)
   {
      mGuiModelSelections.push_back({ TranslatableString{ wxString(m), {}} });
   }

   mSupportedModes = { "transcribe", "translate" };

   for (auto m : mSupportedModes)
   {
      mGuiModeSelections.push_back({ TranslatableString{ wxString(m), {}} });
   }

   mSupportedLanguages.push_back("auto");
   for (auto &e : g_lang_entries)
   {
      mSupportedLanguages.push_back(e.lang_full_string);
   }

   for (auto &l : mSupportedLanguages)
   {
      mGuiLanguageSelections.push_back({ TranslatableString{ wxString(l), {}} });
   }
}

EffectOVWhisperTranscriptionGenAI::~EffectOVWhisperTranscriptionGenAI()
{
}

ComponentInterfaceSymbol EffectOVWhisperTranscriptionGenAI::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVWhisperTranscriptionGenAI::GetDescription() const
{
   return XO("Creates a label track with transcribed text from spoken audio");
}

VendorSymbol EffectOVWhisperTranscriptionGenAI::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

EffectType EffectOVWhisperTranscriptionGenAI::GetType() const
{
   return EffectTypeAnalyze;
}

bool EffectOVWhisperTranscriptionGenAI::Process(EffectInstance&, EffectSettings&)
{
   try
   {
      mIsCancelled = false;

      if (mSupportedModels.empty())
      {
         EffectUIServices::DoMessageBox(*this,
            XO("Whisper Transcription failed. There doesn't seem to be any whisper models installed."),
            wxICON_STOP,
            XO("Error"));
         return false;
      }

      
      //  two label tracks, making the transition at each speaker turn.
      auto whisper_model_variant = mSupportedModels[m_modelSelectionChoice];

      // Do not use mWaveTracks here.  We will possibly DELETE tracks,
      // so we must use the "real" tracklist.
      EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
      bool bGoodResult = true;

      // Determine the total time (in samples) used by all of the target tracks
      sampleCount totalTime = 0;

      //maybe this needs to be in the trackRange loop? The behavior right now is that it will
      // iterate through all selected tracks and dump all of the results in the same label
      // track... which isn't great.
      std::shared_ptr<AddedAnalysisTrack> addedTrack0;
      std::shared_ptr<AddedAnalysisTrack> addedTrack1;

      LabelTrack* lt0{};
      std::string label_track_name = "Transcription";

      label_track_name = label_track_name + "(" + whisper_model_variant + ")";
      addedTrack0 = (MyAddAnalysisTrack(*this, label_track_name)), lt0 = addedTrack0->get();

      //resample all tracks to 16khz
      for (auto pOutWaveTrack : outputs.Get().Selected<WaveTrack>())
      {
         auto rate = pOutWaveTrack->GetRate();
         if (rate != 16000)
         {
            mProgress->SetMessage(XO("Resampling to 16khz"));
            pOutWaveTrack->Resample(16000, mProgress);
         }
      }

      sampleCount curTime = 0;

      // Convert stereo tracks to mono
      for (auto pOutWaveTrack : outputs.Get().Selected<WaveTrack>())
      {
         if (pOutWaveTrack->Channels().size() > 1)
         {
            auto start = pOutWaveTrack->TimeToLongSamples(pOutWaveTrack->GetStartTime());
            auto end = pOutWaveTrack->TimeToLongSamples(pOutWaveTrack->GetEndTime());

            totalTime += (end - start);

            bGoodResult = ProcessStereoToMono(curTime, totalTime, *pOutWaveTrack);
            if (!bGoodResult)
            {
               return false;
            }
         }
      }

      mProgress->SetMessage(XO("Running Whisper Transcription using OpenVINO"));

      //finally, run whisper
      for (auto pOutWaveTrack : outputs.Get().Selected<WaveTrack>())
      {
         bGoodResult = ProcessWhisper(pOutWaveTrack, lt0);

         if (!bGoodResult || mIsCancelled)
            break;
      }

      //we intentionally do not commit. We made changes (sample rate, stereo-to-mono) to the input tracks that
      // we want to throw away.
      //outputs.Commit();

      if (bGoodResult && !mIsCancelled)
      {
         // No cancellation, so commit the addition of the track.
         addedTrack0->Commit();
         if (addedTrack1)
            addedTrack1->Commit();
      }

      return bGoodResult;
   }
   catch (const std::exception& error) {
      wxLogError("In Whisper Transcription Effect, exception: %s", error.what());
      EffectUIServices::DoMessageBox(*this,
         XO("Whisper Transcription failed. See details in Help->Diagnostics->Show Log..."),
         wxICON_STOP,
         XO("Error"));
   }

   return false;
}

bool EffectOVWhisperTranscriptionGenAI::ProcessStereoToMono(sampleCount& curTime, sampleCount totalTime, WaveTrack& track)
{
   auto idealBlockLen = track.GetMaxBlockSize() * 2;
   bool bResult = true;
   sampleCount processed = 0;

   const auto start = track.GetStartTime();
   const auto end = track.GetEndTime();

   Mixer::Inputs tracks;
   tracks.emplace_back(
      track.SharedPointer<const SampleTrack>(), GetEffectStages(track));

   Mixer mixer(
      move(tracks), std::nullopt,
      true, // Throw to abort mix-and-render if read fails:
      Mixer::WarpOptions{ inputTracks()->GetOwner() }, start, end, 1,
      idealBlockLen,
      false, // Not interleaved
      track.GetRate(), floatSample);

   // Always make mono output; don't use EmptyCopy
   auto outTrack = track.EmptyCopy(1);
   auto tempList = TrackList::Temporary(nullptr, outTrack);
   outTrack->ConvertToSampleFormat(floatSample);

   double denominator = track.GetChannelVolume(0) + track.GetChannelVolume(1);
   while (auto blockLen = mixer.Process()) {
      auto buffer = mixer.GetBuffer();
      for (auto i = 0; i < blockLen; i++)
         ((float*)buffer)[i] /= denominator;

      // If mixing channels that both had only 16 bit effective format
      // (for example), and no gains or envelopes, still there should be
      // dithering because of the averaging above, which may introduce samples
      // lying between the quantization levels.  So use widestSampleFormat.
      outTrack->Append(0,
         buffer, floatSample, blockLen, 1, widestSampleFormat);

      curTime += blockLen;
      if (TotalProgress(curTime.as_double() / totalTime.as_double()))
         return false;
   }
   outTrack->Flush();

   track.Clear(start, end);
   track.Paste(start, *outTrack);

   return bResult;
}

bool EffectOVWhisperTranscriptionGenAI::ProcessWhisper(WaveTrack* mono, LabelTrack* lt0)
{
   double trackStart = mono->GetStartTime();
   double trackEnd = mono->GetEndTime();

   // Set the current bounds to whichever left marker is
   // greater and whichever right marker is less:
   const double curT0 = std::max(trackStart, mT0);
   const double curT1 = std::min(trackEnd, mT1);

   std::cout << "curT0 = " << curT0 << std::endl;
   std::cout << "curT1 = " << curT1 << std::endl;
   double label_time_offset = mT0 < curT0 ? curT0 : mT0;

   bool ret = true;
   if (curT1 > curT0) {
      auto start = mono->TimeToLongSamples(curT0);
      auto end = mono->TimeToLongSamples(curT1);

      //Get the length of the buffer (as double). len is
      //used simple to calculate a progress meter, so it is easier
      //to make it a double now than it is to do it later
      auto len = (end - start).as_double();

      size_t total_samples = (end - start).as_size_t();
      std::vector<float> mono_samples;
      mono_samples.resize(total_samples);

      float* buf[1] = { mono_samples.data() };
      bool bOkay = mono->GetFloats(0, 1, buf, start, total_samples);
      if (!bOkay)
      {
         throw std::runtime_error("unable to get all mono samples. GetFloats() failed for " +
            std::to_string(total_samples) + " samples");
      }

      ret = Whisper(mono_samples, lt0, label_time_offset);
   }
   else
   {
      throw std::runtime_error("unexpected case encountered where curT0 (" + std::to_string(curT0) +
         ") <= curT1(" + std::to_string(curT1) + ")");
   }

   return ret;
}

bool EffectOVWhisperTranscriptionGenAI::UpdateProgress(double perc)
{
   std::lock_guard<std::mutex> guard(mMutex);
   mProgressFrac = perc;

   return mIsCancelled;
}

class CustomWhisperStreamer : public ov::genai::StreamerBase
{
public: 
   CustomWhisperStreamer(EffectOVWhisperTranscriptionGenAI *effect, size_t total_samples)
   : _effect(effect) {

      //first calculate the total time in seconds
      size_t total_time = total_samples / 16000;
      if ((total_time % 30) != 0)
      {
         total_time = ((total_time / 30) + 1) * 30;
      }

      _total_expected_callbacks = (total_time / 30) + 1;

   }
   ov::genai::StreamingStatus write(int64_t token) {
      return ov::genai::StreamingStatus::RUNNING;
   }

   ov::genai::StreamingStatus write(const std::vector<int64_t>& tokens) {
      _n_write_calls++;
      double perc_complete = static_cast<double>(_n_write_calls) / static_cast<double>(_total_expected_callbacks);

      if (perc_complete > 1) {
         perc_complete = 1;
      }

      auto bCancelled = _effect->UpdateProgress(perc_complete);
      if (bCancelled) {
         return ov::genai::StreamingStatus::CANCEL;
      }
      return ov::genai::StreamingStatus::RUNNING;
   }

   virtual void end() {
   }

private:
   EffectOVWhisperTranscriptionGenAI* _effect;
   int _total_expected_callbacks;
   int _n_write_calls = 0;

};

bool EffectOVWhisperTranscriptionGenAI::Whisper(std::vector<float>& mono_samples, LabelTrack* lt0, double start_time)
{
   bool ret = true;

   std::string device_name = mSupportedDevices[m_deviceSelectionChoice];
   std::cout << "Creating ov::genai::WhisperPipeline with device = " << device_name << std::endl;

   std::string ui_whisper_model_name = mSupportedModels[m_modelSelectionChoice];
   auto whisper_model_path = _ui_name_to_model_info[ui_whisper_model_name].folderpath;
   std::cout << "whisper_model_path = " << whisper_model_path << std::endl;

   FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());

   //Note: Using a variant of wstring conversion that seems to work more reliably when there are special characters present in the path.
   std::string cache_path = wstring_to_string(wxFileName(cache_folder).GetFullPath().ToStdWstring());

   std::shared_ptr< ov::genai::WhisperPipeline > pipeline;
   std::cout << "Setting cache_dir to " << cache_path << std::endl;
   ov::AnyMap properties = { ov::cache_dir(cache_path) };

   {
      std::future_status status;
      float total_time = 0.f;

      auto init_whisper_fut = std::async(std::launch::async, [&whisper_model_path, &device_name, &properties]() {
         OVLocaleWorkaround wa;
         return std::make_shared< ov::genai::WhisperPipeline >(whisper_model_path, device_name, properties);
         });

      do {
         using namespace std::chrono_literals;
         status = init_whisper_fut.wait_for(0.5s);
         {
            std::string message = "Loading Whisper AI Model for " + device_name + "...";
            if (total_time > 10)
            {
               message += " (This could take a while if this is the first time running this feature with this device)";
            }
            TotalProgress(0.01, TranslatableString{ wxString(message), {} });
         }

         total_time += 0.5;

      } while (status != std::future_status::ready);

      pipeline = init_whisper_fut.get();
   }

   if (!pipeline)
   {
      throw std::runtime_error("Whisper GenAI creation / initialization failed");
   }

   ov::genai::WhisperGenerationConfig config = pipeline->get_generation_config();
   config.max_new_tokens = SIZE_MAX;

   if (config.is_multilingual)
   {
      std::string slang = mSupportedLanguages[m_languageSelectionChoice];
      if (slang == "auto")
      {
         config.language = std::optional<std::string>{};

         std::cout << "config.language = {} (auto)" << std::endl;
      }
      else
      {
         bool bFound = false;
         for (auto& e : g_lang_entries)
         {
            if (slang == e.lang_full_string)
            {
               bFound = true;
               config.language = e.lang_code_string;
               break;
            }
         }

         if (!bFound)
         {
            throw std::runtime_error("Invalid language selection!");
         }

         std::cout << "config.language = " << *config.language << std::endl;
      }

      std::string smode = mSupportedModes[m_modeSelectionChoice];
      config.task = smode;
      std::cout << "config.task = " << *config.task << std::endl;
   }

   config.return_timestamps = true;

   std::shared_ptr< CustomWhisperStreamer > streamer = std::make_shared< CustomWhisperStreamer>(this, mono_samples.size());

   mProgressFrac = 0.0;
   mProgMessage = "Running Whisper Transcription using OpenVINO";

   auto whisper_parallel_run_future = std::async(std::launch::async,
      [this, &pipeline, &mono_samples, &config, &streamer]
      {
         OVLocaleWorkaround wa;
         return pipeline->generate(mono_samples, config, streamer);
      }
   );

   std::future_status status;
   do {
      using namespace std::chrono_literals;
      status = whisper_parallel_run_future.wait_for(0.5s);
      {
         std::lock_guard<std::mutex> guard(mMutex);
         mProgress->SetMessage(TranslatableString{ wxString(mProgMessage), {} });
         if (TotalProgress(mProgressFrac))
         {
            mIsCancelled = true;
         }
      }

   } while (status != std::future_status::ready);

   auto result = whisper_parallel_run_future.get();

   for (auto& chunk : *result.chunks) {
      double start = chunk.start_ts + start_time;
      double end = chunk.end_ts + start_time;
      auto wxText = wxString::FromUTF8(chunk.text);
      lt0->AddLabel(SelectedRegion(start, end), wxText);
   }

   return ret;
}

void EffectOVWhisperTranscriptionGenAI::show_or_hide_advanced_options()
{
   if (advancedSizer)
   {
      advancedSizer->ShowItems(mbAdvanced);
      advancedSizer->Layout();
   }
}

void EffectOVWhisperTranscriptionGenAI::OnAdvancedCheckboxChanged(wxCommandEvent& evt)
{
   mbAdvanced = mShowAdvancedOptionsCheckbox->GetValue();

   show_or_hide_advanced_options();

   if (mUIParent)
   {
      mUIParent->Layout();
      mUIParent->SetMinSize(mUIParent->GetSizer()->GetMinSize());
      mUIParent->SetSize(mUIParent->GetSizer()->GetMinSize());
      mUIParent->Fit();

      auto p = mUIParent->GetParent();
      if (p)
      {
         p->Fit();
      }

   }
}

void EffectOVWhisperTranscriptionGenAI::OnDeviceInfoButtonClicked(wxCommandEvent& evt)
{
   std::string device_mapping_str = "";
   for (auto e : m_simple_to_full_device_map)
   {
      device_mapping_str += e.first + " = " + e.second;
      device_mapping_str += "\n";
   }
   auto v = TranslatableString(device_mapping_str, {});

   EffectUIServices::DoMessageBox(*this,
      v,
      wxICON_INFORMATION,
      XO("OpenVINO Device Details"));
}

bool EffectOVWhisperTranscriptionGenAI::TransferDataToWindow(const EffectSettings&)
{
   if (!mUIParent || !mUIParent->TransferDataToWindow())
   {
      return false;
   }


   if (mSupportedModels.empty())
   {
      wxLogInfo("OpenVINO Whisper Transcription has no models installed.");
      EffectEditor::EnableApply(mUIParent, false);
   }

   return true;
}

bool EffectOVWhisperTranscriptionGenAI::TransferDataFromWindow(EffectSettings&)
{
   if (!mUIParent || !mUIParent->Validate() || !mUIParent->TransferDataFromWindow())
   {
      return false;
   }

   mInitialPrompt = audacity::ToUTF8(mInitialPromptCtrl->GetLineText(0));

   return true;
}

std::unique_ptr<EffectEditor> EffectOVWhisperTranscriptionGenAI::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess& access,
   const EffectOutputs*)
{
   mUIParent = S.GetParent();

   S.AddSpace(0, 5);
   S.StartVerticalLay();
   {
      S.StartStatic(XO(""), wxLEFT);
      {
         S.StartMultiColumn(4, wxEXPAND);
         {
            mTypeChoiceDeviceCtrl = S.Id(ID_Type)
               .MinSize({ -1, -1 })
               .Validator<wxGenericValidator>(&m_deviceSelectionChoice)
               .AddChoice(XXO("OpenVINO Inference Device:"),
                  Msgids(mGuiDeviceSelections.data(), mGuiDeviceSelections.size()));
            S.AddVariableText(XO(""));

            auto device_info_button = S.Id(ID_Type_DeviceInfoButton).AddButton(XO("Device Details..."));

            S.SetStretchyCol(2);
         }
         S.EndMultiColumn();
      }
      S.EndStatic();

      S.StartMultiColumn(4, wxLEFT);
      {
         //m_deviceSelectionChoice
         mTypeChoiceModelCtrl = S.Id(ID_Type_Model)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_modelSelectionChoice)
            .AddChoice(XXO("Whisper Model:"),
               Msgids(mGuiModelSelections.data(), mGuiModelSelections.size()));
      }
      S.EndMultiColumn();
  
      S.StartMultiColumn(4, wxLEFT);
      {
         //m_deviceSelectionChoice
         mTypeChoiceModelCtrl = S.Id(ID_Type_Mode)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_modeSelectionChoice)
            .AddChoice(XXO("Mode:"),
               Msgids(mGuiModeSelections.data(), mGuiModeSelections.size()));

         mTypeChoiceModelCtrl = S.Id(ID_Type_Language)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_languageSelectionChoice)
            .AddChoice(XXO("Source Language:"),
               Msgids(mGuiLanguageSelections.data(), mGuiLanguageSelections.size()));
      }
      S.EndMultiColumn();

      //advanced options
      S.StartMultiColumn(2, wxLEFT);
      {
         mShowAdvancedOptionsCheckbox = S.Id(ID_Type_AdvancedCheckbox).AddCheckBox(XXO("&Advanced Options"), mbAdvanced);
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         mInitialPromptCtrl = S.Style(wxTE_LEFT)
            .AddTextBox(XXO("Initial Prompt:"), wxString(mInitialPrompt), 30);

         advancedSizer = mInitialPromptCtrl->GetContainingSizer();

         mMaxTextSegLengthCtrl = S.Name(XO("Max Segment Length"))
            .Validator<IntegerValidator<int>>(&mMaxTextSegLength,
               NumValidatorStyle::DEFAULT,
               0,
               1000)
            .AddTextBox(XO("Max Segment Length"), L"", 12);

         mBeamSizeCtrl = S.Name(XO("Beam Size"))
            .Validator<IntegerValidator<int>>(&mBeamSize,
               NumValidatorStyle::DEFAULT,
               1,
               1000)
            .AddTextBox(XO("Beam Size"), L"", 12);

         mBestOfCtrl = S.Name(XO("Best Of"))
            .Validator<IntegerValidator<int>>(&mBestOf,
               NumValidatorStyle::DEFAULT,
               1,
               1000)
            .AddTextBox(XO("Best Of"), L"", 12);

      }
      S.EndMultiColumn();

   }
   S.EndVerticalLay();

   show_or_hide_advanced_options();

   return nullptr;
}

