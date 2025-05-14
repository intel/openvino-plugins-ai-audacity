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

struct whispercpp_lang_entry
{
   std::string lang_full_string;
   std::string lang_code_string;
   int id;
};

static const std::vector< whispercpp_lang_entry > g_lang_entries =
{
    { "english", "en", 0, },
    { "afrikaans", "af", 68, },
    { "albanian", "sq", 58, },
    { "amharic", "am", 75, },
    { "arabic", "ar", 13, },
    { "armenian", "hy", 53, },
    { "assamese", "as", 91, },
    { "azerbaijani", "az", 45, },
    { "bashkir", "ba", 96, },
    { "basque", "eu", 51, },
    { "belarusian", "be", 71, },
    { "bengali", "bn", 43, },
    { "bosnian", "bs", 56, },
    { "breton", "br", 50, },
    { "bulgarian", "bg", 33, },
    { "catalan", "ca", 11, },
    { "chinese", "zh", 1, },
    { "croatian", "hr", 32, },
    { "czech", "cs", 24, },
    { "danish", "da", 26, },
    { "dutch", "nl", 12, },
    { "estonian", "et", 48, },
    { "faroese", "fo", 79, },
    { "finnish", "fi", 18, },
    { "french", "fr", 6, },
    { "galician", "gl", 60, },
    { "georgian", "ka", 70, },
    { "german", "de", 2, },
    { "greek", "el", 22, },
    { "gujarati", "gu", 74, },
    { "haitian creole", "ht", 80, },
    { "hausa", "ha", 95, },
    { "hawaiian", "haw", 93, },
    { "hebrew", "he", 20, },
    { "hindi", "hi", 17, },
    { "hungarian", "hu", 27, },
    { "icelandic", "is", 52, },
    { "indonesian", "id", 16, },
    { "italian", "it", 15, },
    { "japanese", "ja", 7, },
    { "javanese", "jw", 97, },
    { "kannada", "kn", 47, },
    { "kazakh", "kk", 57, },
    { "khmer", "km", 64, },
    { "korean", "ko", 5, },
    { "lao", "lo", 77, },
    { "latin", "la", 35, },
    { "latvian", "lv", 42, },
    { "lingala", "ln", 94, },
    { "lithuanian", "lt", 34, },
    { "luxembourgish", "lb", 86, },
    { "macedonian", "mk", 49, },
    { "malagasy", "mg", 90, },
    { "malay", "ms", 23, },
    { "malayalam", "ml", 37, },
    { "maltese", "mt", 84, },
    { "maori", "mi", 36, },
    { "marathi", "mr", 61, },
    { "mongolian", "mn", 55, },
    { "myanmar", "my", 87, },
    { "nepali", "ne", 54, },
    { "norwegian", "no", 29, },
    { "nynorsk", "nn", 83, },
    { "occitan", "oc", 69, },
    { "pashto", "ps", 81, },
    { "persian", "fa", 41, },
    { "polish", "pl", 10, },
    { "portuguese", "pt", 8, },
    { "punjabi", "pa", 62, },
    { "romanian", "ro", 25, },
    { "russian", "ru", 4, },
    { "sanskrit", "sa", 85, },
    { "serbian", "sr", 44, },
    { "shona", "sn", 65, },
    { "sindhi", "sd", 73, },
    { "sinhala", "si", 63, },
    { "slovak", "sk", 39, },
    { "slovenian", "sl", 46, },
    { "somali", "so", 67, },
    { "spanish", "es", 3, },
    { "sundanese", "su", 98, },
    { "swahili", "sw", 59, },
    { "swedish", "sv", 14, },
    { "tagalog", "tl", 89, },
    { "tajik", "tg", 72, },
    { "tamil", "ta", 28, },
    { "tatar", "tt", 92, },
    { "telugu", "te", 40, },
    { "thai", "th", 30, },
    { "tibetan", "bo", 88, },
    { "turkish", "tr", 9, },
    { "turkmen", "tk", 82, },
    { "ukrainian", "uk", 21, },
    { "urdu", "ur", 31, },
    { "uzbek", "uz", 78, },
    { "vietnamese", "vi", 19, },
    { "welsh", "cy", 38, },
    { "yiddish", "yi", 76, },
    { "yoruba", "yo", 66, },
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

//Given a whisper basename (i.e. 'base', 'small', etc.), check to make sure that all
// of the required model files exist in the right location, and return true / false.
// true: All the files exist!
// false: At least one of the required files are missing.
static bool is_whisper_model_present(std::string whisper_basename)
{
   std::cout << "is_whisper_model_present(" << whisper_basename << ")" << std::endl;
   auto model_folder = wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath();

   {
      std::string ggml_binname = std::string("ggml-") + whisper_basename + std::string(".bin");
      auto whisper_ggml_model_path = wxFileName(model_folder, wxString(ggml_binname));
      if (!whisper_ggml_model_path.FileExists())
      {
         std::cout << "is_whisper_model_present: returning false because " << ggml_binname << " doesn't exist." << std::endl;
         return false;
      }
   }

   auto ov_model_basename = std::string("ggml-") + whisper_basename + std::string("-encoder-openvino");
   {
      std::string whisper_openvino_xml_file = ov_model_basename + std::string(".xml");
      auto whisper_openvino_model_xml_path = wxFileName(model_folder, wxString(whisper_openvino_xml_file));
      if (!whisper_openvino_model_xml_path.FileExists())
      {
         std::cout << "is_whisper_model_present: returning false because " << whisper_openvino_xml_file << " doesn't exist." << std::endl;
         return false;
      }
   }

   {
      std::string whisper_openvino_bin_file = ov_model_basename + std::string(".bin");
      auto whisper_openvino_model_bin_path = wxFileName(model_folder, wxString(whisper_openvino_bin_file));
      if (!whisper_openvino_model_bin_path.FileExists())
      {
         std::cout << "is_whisper_model_present: returning false because " << whisper_openvino_bin_file << " doesn't exist." << std::endl;
         return false;
      }
   }

   std::cout << "is_whisper_model_present: returning true" << std::endl;
   return true;
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

   std::vector<std::string> possible_supported_models =
   { "tiny", "tiny.en",
      "base", "base.en",
      "small", "small.en", "small.en-tdrz",
      "medium", "medium.en",
      "large-v1", "large-v2", "large-v3"
   };

   //For each possible model, check to see if the required model files
   // (ggml bin, openvino IR's) are present in the 'openvino-models' dir.
   // If so, then populate our drop down list with this model so that it
   // is selectable.
   for (auto m : possible_supported_models)
   {
      if (is_whisper_model_present(m))
      {
         mSupportedModels.push_back(m);
         mGuiModelSelections.push_back({ TranslatableString{ wxString(m), {}} });
      }
   }

   mSupportedModes = { "transcribe", "translate" };

   for (auto m : mSupportedModes)
   {
      mGuiModeSelections.push_back({ TranslatableString{ wxString(m), {}} });
   }

   mSupportedLanguages.push_back("auto");
   for (auto e : g_lang_entries)
   {
      mSupportedLanguages.push_back(e.lang_full_string);
   }

   for (auto l : mSupportedLanguages)
   {
      mGuiLanguageSelections.push_back({ TranslatableString{ wxString(l), {}} });
   }

   mBeamSize = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
   mBestOf = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
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

      // if user has installed / enabled specific tdrz model, then we will alternate between
      // two label tracks.
      bool btdrz = (whisper_model_variant  == "small.en-tdrz");

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
      if (btdrz)
      {
         label_track_name = "Transcription 0";
      }

      label_track_name = label_track_name + "(" + whisper_model_variant + ")";
      addedTrack0 = (MyAddAnalysisTrack(*this, label_track_name)), lt0 = addedTrack0->get();

      LabelTrack* lt1{};
      if (btdrz)
      {
         label_track_name = "Transcription 1 (" + whisper_model_variant + ")";
         addedTrack1 = (MyAddAnalysisTrack(*this, label_track_name)), lt1 = addedTrack1->get();
      }


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
         bGoodResult = ProcessWhisper(pOutWaveTrack, lt0, lt1);

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

bool EffectOVWhisperTranscriptionGenAI::ProcessWhisper(WaveTrack* mono, LabelTrack* lt0, LabelTrack* lt1)
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

      ret = Whisper(mono_samples, lt0, lt1, label_time_offset);
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
   mProgressFrac = perc / 100.0;

   return true;
}

//  500 -> 00:05.000
// 6000 -> 01:00.000
static std::string to_timestamp(int64_t t, bool comma = false) {
   int64_t msec = t * 10;
   int64_t hr = msec / (1000 * 60 * 60);
   msec = msec - hr * (1000 * 60 * 60);
   int64_t min = msec / (1000 * 60);
   msec = msec - min * (1000 * 60);
   int64_t sec = msec / 1000;
   msec = msec - sec * 1000;

   char buf[32];
   snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int)hr, (int)min, (int)sec, comma ? "," : ".", (int)msec);

   return std::string(buf);
}

static int timestamp_to_sample(int64_t t, int n_samples) {
   return std::max(0, std::min((int)n_samples - 1, (int)((t * WHISPER_SAMPLE_RATE) / 100)));
}



bool EffectOVWhisperTranscriptionGenAI::Whisper(std::vector<float>& mono_samples, LabelTrack* lt0, LabelTrack* lt1, double start_time)
{
   std::cout << "Whisper..." << std::endl;
   bool ret = true;

   std::string device_name = mSupportedDevices[m_deviceSelectionChoice];
   std::cout << "Creating ov::genai::WhisperPipeline with device = " << device_name << std::endl;

   //TODO: duh.
   auto whisper_model_path = "C:\\Users\\LNL\\Workspace\\git\\test\\ai_tabletop_adventure_assistant\\models\\whisper-base";


   FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());

   //Note: Using a variant of wstring conversion that seems to work more reliably when there are special characters present in the path.
   std::string cache_path = wstring_to_string(wxFileName(cache_folder).GetFullPath().ToStdWstring());

   std::shared_ptr< ov::genai::WhisperPipeline > pipeline;
   {
      std::future_status status;
      float total_time = 0.f;

      auto init_whisper_fut = std::async(std::launch::async, [&whisper_model_path, &device_name, &cache_path]() {
         return std::make_shared< ov::genai::WhisperPipeline >(whisper_model_path, device_name);
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
   config.max_new_tokens = SIZE_MAX;  // increase this based on your speech length
   // 'task' and 'language' parameters are supported for multilingual models only
   config.language = "<|en|>";  
   config.task = "transcribe";
   config.return_timestamps = true;

   auto result = pipeline->generate(mono_samples, config);

   std::cout << std::fixed << std::setprecision(2);
   for (auto& chunk : *result.chunks) {
      //std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text << "\n";

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

