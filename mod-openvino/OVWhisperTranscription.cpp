// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "OVWhisperTranscription.h"
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

#include "effects/AnalysisTracks.h"
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

static const std::map<std::string, std::pair<int, std::string>> g_lang = {
    { "en",  { 0,  "english",         } },
    { "zh",  { 1,  "chinese",         } },
    { "de",  { 2,  "german",          } },
    { "es",  { 3,  "spanish",         } },
    { "ru",  { 4,  "russian",         } },
    { "ko",  { 5,  "korean",          } },
    { "fr",  { 6,  "french",          } },
    { "ja",  { 7,  "japanese",        } },
    { "pt",  { 8,  "portuguese",      } },
    { "tr",  { 9,  "turkish",         } },
    { "pl",  { 10, "polish",          } },
    { "ca",  { 11,  "catalan",        } },
    { "nl",  { 12,  "dutch",          } },
    { "ar",  { 13,  "arabic",         } },
    { "sv",  { 14,  "swedish",        } },
    { "it",  { 15,  "italian",        } },
    { "id",  { 16,  "indonesian",     } },
    { "hi",  { 17,  "hindi",          } },
    { "fi",  { 18,  "finnish",        } },
    { "vi",  { 19,  "vietnamese",     } },
    { "he",  { 20,  "hebrew",         } },
    { "uk",  { 21,  "ukrainian",      } },
    { "el",  { 22,  "greek",          } },
    { "ms",  { 23,  "malay",          } },
    { "cs",  { 24,  "czech",          } },
    { "ro",  { 25,  "romanian",       } },
    { "da",  { 26,  "danish",         } },
    { "hu",  { 27,  "hungarian",      } },
    { "ta",  { 28,  "tamil",          } },
    { "no",  { 29,  "norwegian",      } },
    { "th",  { 30,  "thai",           } },
    { "ur",  { 31,  "urdu",           } },
    { "hr",  { 32,  "croatian",       } },
    { "bg",  { 33,  "bulgarian",      } },
    { "lt",  { 34,  "lithuanian",     } },
    { "la",  { 35,  "latin",          } },
    { "mi",  { 36,  "maori",          } },
    { "ml",  { 37,  "malayalam",      } },
    { "cy",  { 38,  "welsh",          } },
    { "sk",  { 39,  "slovak",         } },
    { "te",  { 40,  "telugu",         } },
    { "fa",  { 41,  "persian",        } },
    { "lv",  { 42,  "latvian",        } },
    { "bn",  { 43,  "bengali",        } },
    { "sr",  { 44,  "serbian",        } },
    { "az",  { 45,  "azerbaijani",    } },
    { "sl",  { 46,  "slovenian",      } },
    { "kn",  { 47,  "kannada",        } },
    { "et",  { 48,  "estonian",       } },
    { "mk",  { 49,  "macedonian",     } },
    { "br",  { 50,  "breton",         } },
    { "eu",  { 51,  "basque",         } },
    { "is",  { 52,  "icelandic",      } },
    { "hy",  { 53,  "armenian",       } },
    { "ne",  { 54,  "nepali",         } },
    { "mn",  { 55,  "mongolian",      } },
    { "bs",  { 56,  "bosnian",        } },
    { "kk",  { 57,  "kazakh",         } },
    { "sq",  { 58,  "albanian",       } },
    { "sw",  { 59,  "swahili",        } },
    { "gl",  { 60,  "galician",       } },
    { "mr",  { 61,  "marathi",        } },
    { "pa",  { 62,  "punjabi",        } },
    { "si",  { 63,  "sinhala",        } },
    { "km",  { 64,  "khmer",          } },
    { "sn",  { 65,  "shona",          } },
    { "yo",  { 66,  "yoruba",         } },
    { "so",  { 67,  "somali",         } },
    { "af",  { 68,  "afrikaans",      } },
    { "oc",  { 69,  "occitan",        } },
    { "ka",  { 70,  "georgian",       } },
    { "be",  { 71,  "belarusian",     } },
    { "tg",  { 72,  "tajik",          } },
    { "sd",  { 73,  "sindhi",         } },
    { "gu",  { 74,  "gujarati",       } },
    { "am",  { 75,  "amharic",        } },
    { "yi",  { 76,  "yiddish",        } },
    { "lo",  { 77,  "lao",            } },
    { "uz",  { 78,  "uzbek",          } },
    { "fo",  { 79,  "faroese",        } },
    { "ht",  { 80,  "haitian creole", } },
    { "ps",  { 81,  "pashto",         } },
    { "tk",  { 82,  "turkmen",        } },
    { "nn",  { 83,  "nynorsk",        } },
    { "mt",  { 84,  "maltese",        } },
    { "sa",  { 85,  "sanskrit",       } },
    { "lb",  { 86,  "luxembourgish",  } },
    { "my",  { 87,  "myanmar",        } },
    { "bo",  { 88,  "tibetan",        } },
    { "tl",  { 89,  "tagalog",        } },
    { "mg",  { 90,  "malagasy",       } },
    { "as",  { 91,  "assamese",       } },
    { "tt",  { 92,  "tatar",          } },
    { "haw", { 93,  "hawaiian",       } },
    { "ln",  { 94,  "lingala",        } },
    { "ha",  { 95,  "hausa",          } },
    { "ba",  { 96,  "bashkir",        } },
    { "jw",  { 97,  "javanese",       } },
    { "su",  { 98,  "sundanese",      } },
};

#if 0
const EffectParameterMethods& EffectOVWhisperTranscription::Parameters() const
{
   static CapturedParameters<EffectOVWhisperTranscription,
      Start, Stop
   > parameters;
   return parameters;
}
#endif

const ComponentInterfaceSymbol EffectOVWhisperTranscription::Symbol
{ XO("OpenVINO Whisper Transcription") };

namespace { BuiltinEffectsModule::Registration< EffectOVWhisperTranscription > reg; }

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


EffectOVWhisperTranscription::EffectOVWhisperTranscription()
{
   ov::Core core;
   auto devices = core.get_available_devices();

   for (auto d : devices)
   {
      //GNA devices are not supported
      if (d.find("GNA") != std::string::npos) continue;

      if (d == "VPU") continue;

      mSupportedDevices.push_back(d);
   }

   for (auto d : mSupportedDevices)
   {
      mGuiDeviceSelections.push_back({ TranslatableString{ wxString(d), {}} });
   }

   mSupportedModels = { "base" };

   for (auto m : mSupportedModels)
   {
      mGuiModelSelections.push_back({ TranslatableString{ wxString(m), {}} });
   }

   mSupportedModes = { "transcribe", "translate" };

   for (auto m : mSupportedModes)
   {
      mGuiModeSelections.push_back({ TranslatableString{ wxString(m), {}} });
   }

   mSupportedLanguages.push_back("auto");
   for (auto e : g_lang)
   {
      mSupportedLanguages.push_back(e.second.second);
   }

   for (auto l : mSupportedLanguages)
   {
      mGuiLanguageSelections.push_back({ TranslatableString{ wxString(l), {}} });
   }
}

EffectOVWhisperTranscription::~EffectOVWhisperTranscription()
{
}

ComponentInterfaceSymbol EffectOVWhisperTranscription::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVWhisperTranscription::GetDescription() const
{
   return XO("Creates a label track with transcribed text from spoken audio");
}

VendorSymbol EffectOVWhisperTranscription::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

EffectType EffectOVWhisperTranscription::GetType() const
{
   return EffectTypeAnalyze;
}

bool EffectOVWhisperTranscription::Process(EffectInstance&, EffectSettings&)
{
   try
   {
      mIsCancelled = false;

      // Do not use mWaveTracks here.  We will possibly DELETE tracks,
      // so we must use the "real" tracklist.
      EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
      bool bGoodResult = true;

      // Determine the total time (in samples) used by all of the target tracks
      sampleCount totalTime = 0;

      //maybe this needs to be in the trackRange loop?
      std::shared_ptr<AddedAnalysisTrack> addedTrack;
      std::optional<ModifiedAnalysisTrack> modifiedTrack;
      const wxString name{ _("Transcription") };

      auto clt = *inputTracks()->Any< const LabelTrack >().find_if(
         [&](const Track* track) { return track->GetName() == name; });

      LabelTrack* lt{};
      if (!clt)
      {
         addedTrack = (MyAddAnalysisTrack(*this, name)), lt = addedTrack->get();
      }
      else
      {
         modifiedTrack.emplace(MyModifyAnalysisTrack(*this, *clt, name)),
            lt = modifiedTrack->get();
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
         bGoodResult = ProcessWhisper(pOutWaveTrack, lt);

         if (!bGoodResult || mIsCancelled)
            break;
      }

      //we intentionally do not commit. We made changes (sample rate, stereo-to-mono) to the input tracks that
      // we want to throw away.
      //outputs.Commit();

      if (bGoodResult && !mIsCancelled)
      {
         // No cancellation, so commit the addition of the track.
         if (addedTrack)
            addedTrack->Commit();
         if (modifiedTrack)
            modifiedTrack->Commit();
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

bool EffectOVWhisperTranscription::ProcessStereoToMono(sampleCount& curTime, sampleCount totalTime, WaveTrack& track)
{
   auto idealBlockLen = track.GetMaxBlockSize() * 2;
   bool bResult = true;
   sampleCount processed = 0;

   const auto start = track.GetStartTime();
   const auto end = track.GetEndTime();

   Mixer::Inputs tracks;
   tracks.emplace_back(
      track.SharedPointer<const SampleTrack>(), GetEffectStages(track));

   Mixer mixer(move(tracks),
      true,                // Throw to abort mix-and-render if read fails:
      Mixer::WarpOptions{ inputTracks()->GetOwner() },
      start,
      end,
      1,
      idealBlockLen,
      false,               // Not interleaved
      track.GetRate(),
      floatSample);

   // Always make mono output; don't use WideEmptyCopy
   auto outTrack = track.EmptyCopy();
   auto tempList = TrackList::Temporary(nullptr, outTrack, nullptr);
   assert(outTrack->IsLeader());
   outTrack->ConvertToSampleFormat(floatSample);

   double denominator = track.GetChannelGain(0) + track.GetChannelGain(1);
   while (auto blockLen = mixer.Process()) {
      auto buffer = mixer.GetBuffer();
      for (auto i = 0; i < blockLen; i++)
         ((float*)buffer)[i] /= denominator;

      // If mixing channels that both had only 16 bit effective format
      // (for example), and no gains or envelopes, still there should be
      // dithering because of the averaging above, which may introduce samples
      // lying between the quantization levels.  So use widestSampleFormat.
      outTrack->Append(buffer, floatSample, blockLen, 1, widestSampleFormat);

      curTime += blockLen;
      if (TotalProgress(curTime.as_double() / totalTime.as_double()))
         return false;
   }
   outTrack->Flush();

   track.Clear(start, end);
   track.Paste(start, *outTrack);

   return bResult;
}

bool EffectOVWhisperTranscription::ProcessWhisper(WaveTrack* mono, LabelTrack* lt)
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
      bool bOkay = mono->GetFloats(mono_samples.data(), start, total_samples);
      if (!bOkay)
      {
         throw std::runtime_error("unable to get all mono samples. GetFloats() failed for " +
            std::to_string(total_samples) + " samples");
      }

      ret = Whisper(mono_samples, lt, label_time_offset);
   }
   else
   {
      throw std::runtime_error("unexpected case encountered where curT0 (" + std::to_string(curT0) +
         ") <= curT1(" + std::to_string(curT1) + ")");
   }

   return ret;
}

bool EffectOVWhisperTranscription::UpdateProgress(double perc)
{
   // This returns true if user clicks 'cancel'
   if (TotalProgress(perc / 100.0))
   {
      std::lock_guard<std::mutex> guard(mMutex);
      mIsCancelled = true;
      return false;
   }

   return true;
}

struct whisper_params {
   int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
   int32_t n_processors = 1;
   int32_t offset_t_ms = 0;
   int32_t offset_n = 0;
   int32_t duration_ms = 0;
   int32_t max_context = -1;
   int32_t max_len = 0;
   int32_t best_of = 2;
   int32_t beam_size = -1;

   float word_thold = 0.01f;
   float entropy_thold = 2.40f;
   float logprob_thold = -1.00f;

   bool speed_up = false;
   bool translate = false;
   bool detect_language = false;
   bool diarize = false;
   bool split_on_word = false;
   bool no_fallback = false;
   bool output_txt = false;
   bool output_vtt = false;
   bool output_srt = false;
   bool output_wts = false;
   bool output_csv = false;
   bool output_jsn = false;
   bool output_lrc = false;
   bool print_special = false;
   bool print_colors = false;
   bool print_progress = false;
   bool no_timestamps = false;

   std::string language = "en";
   std::string prompt;
   std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
   std::string model = "models/ggml-base.en.bin";

   std::vector<std::string> fname_inp = {};
   std::vector<std::string> fname_out = {};

   LabelTrack* lt = nullptr;
   int64_t nsamples = 0;
   double start_time = 0.;
};

struct whisper_print_user_data {
   const whisper_params* params;

   const std::vector<std::vector<float>>* pcmf32s;
};

// Terminal color map. 10 colors grouped in ranges [0.0, 0.1, ..., 0.9]
// Lowest is red, middle is yellow, highest is green.
const std::vector<std::string> k_colors = {
    "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
    "\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
};

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma = false) {
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

int timestamp_to_sample(int64_t t, int n_samples) {
   return std::max(0, std::min((int)n_samples - 1, (int)((t * WHISPER_SAMPLE_RATE) / 100)));
}

static void whisper_print_segment_callback(struct whisper_context* ctx, struct whisper_state* /*state*/, int n_new, void* user_data) {
   const auto& params = *((whisper_print_user_data*)user_data)->params;
   const auto& pcmf32s = *((whisper_print_user_data*)user_data)->pcmf32s;

   const int n_segments = whisper_full_n_segments(ctx);

   std::string speaker = "";

   int64_t t0 = 0;
   int64_t t1 = 0;

   // print the last n_new segments
   const int s0 = n_segments - n_new;

   if (s0 == 0) {
      //printf("\n");
   }

   for (int i = s0; i < n_segments; i++) {
      if (!params.no_timestamps || params.diarize) {
         t0 = whisper_full_get_segment_t0(ctx, i);
         t1 = whisper_full_get_segment_t1(ctx, i);
      }


      if (params.diarize && pcmf32s.size() == 2) {
         const int64_t n_samples = pcmf32s[0].size();

         const int64_t is0 = timestamp_to_sample(t0, n_samples);
         const int64_t is1 = timestamp_to_sample(t1, n_samples);

         double energy0 = 0.0f;
         double energy1 = 0.0f;

         for (int64_t j = is0; j < is1; j++) {
            energy0 += fabs(pcmf32s[0][j]);
            energy1 += fabs(pcmf32s[1][j]);
         }

         if (energy0 > 1.1 * energy1) {
            speaker = "(speaker 0)";
         }
         else if (energy1 > 1.1 * energy0) {
            speaker = "(speaker 1)";
         }
         else {
            speaker = "(speaker ?)";
         }

         //printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, %s\n", is0, is1, energy0, energy1, speaker.c_str());
      }

      if (params.print_colors) {
         for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
            if (params.print_special == false) {
               const whisper_token id = whisper_full_get_token_id(ctx, i, j);
               if (id >= whisper_token_eot(ctx)) {
                  continue;
               }
            }

            const char* text = whisper_full_get_token_text(ctx, i, j);
            const float  p = whisper_full_get_token_p(ctx, i, j);

            const int col = std::max(0, std::min((int)k_colors.size() - 1, (int)(std::pow(p, 3) * float(k_colors.size()))));

            //printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
         }
      }
      else {
         const char* text = whisper_full_get_segment_text(ctx, i);

        // printf("%s%s", speaker.c_str(), text);


         const int64_t is0 = timestamp_to_sample(t0, params.nsamples);
         const int64_t is1 = timestamp_to_sample(t1, params.nsamples);

         double start = is0 / 16000.0 + params.start_time;
         double end = is1 / 16000.0 + params.start_time;

         auto wxText = wxString::FromUTF8(text);
         params.lt->AddLabel(SelectedRegion(start, end),
            wxText);

      }

      // with timestamps or speakers: each segment on new line
      if (!params.no_timestamps || params.diarize) {
         //printf("\n");
      }

      //fflush(stdout);
   }
}



static void whisper_print_progress_callback(struct whisper_context* ctx, struct whisper_state* /*state*/, int progress, void* user_data)
{
   EffectOVWhisperTranscription* transcription = (EffectOVWhisperTranscription*)user_data;

   if (transcription)
   {
      transcription->UpdateProgress((double)progress);
   }

   //std::cout << "whisper_print_progress_callback: " << progress << std::endl;
}

bool EffectOVWhisperTranscription::_EncoderBegin()
{
   std::lock_guard<std::mutex> guard(mMutex);
   bool ret = !mIsCancelled;
   return ret;
}

//if we return false from this function, it will cancel the whisper processing.
static bool whisper_encoder_callback(struct whisper_context* ctx, struct whisper_state* state, void* user_data)
{
   EffectOVWhisperTranscription* transcription = (EffectOVWhisperTranscription*)user_data;

   return transcription->_EncoderBegin();
}


bool EffectOVWhisperTranscription::Whisper(std::vector<float>& mono_samples, LabelTrack* lt, double start_time)
{
   whisper_params params;
   params.lt = lt;
   params.nsamples = mono_samples.size();
   params.start_time = start_time;

   //whisper init
   FilePath model_folder = FileNames::MkDir(wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath());
   std::string ggml_binname = std::string("ggml-") + mSupportedModels[m_modelSelectionChoice] + std::string(".bin");
   std::string whisper_model_path = audacity::ToUTF8(wxFileName(model_folder, wxString(ggml_binname))
      .GetFullPath());

   FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());

   //Note: Using a variant of wstring conversion that seems to work more reliably when there are special characters present in the path.
   std::string cache_path = wstring_to_string(wxFileName(cache_folder).GetFullPath().ToStdWstring());

   std::string smode = mSupportedModes[m_modeSelectionChoice];
   std::cout << "Mode = " << smode << std::endl;
   params.translate = (smode == "translate");

   std::string slang = mSupportedLanguages[m_languageSelectionChoice];

   if (slang == "auto")
   {
      params.language = "auto";
   }
   else
   {
      params.detect_language = false;
      for (auto e : g_lang)
      {
         if (slang == e.second.second)
         {
            params.language = e.first;
            break;
         }
      }
   }

   std::cout << "lang selection = " << slang << ", params.language = " << params.language << std::endl;

   std::string device_name = mSupportedDevices[m_deviceSelectionChoice];
   std::cout << "Enabling OpenVINO-based Encoder object that will run on " << device_name << std::endl;

   if (m_deviceSelectionChoice >= mSupportedDevices.size())
   {
      std::cout << "Invalid device choice id: " << m_deviceSelectionChoice << std::endl;
      return false;
   }

   struct whisper_context* ctx = nullptr;

   {
      std::future_status status;
      float total_time = 0.f;

      auto init_whisper_fut = std::async(std::launch::async, [&whisper_model_path, &device_name, &cache_path]() {
         struct whisper_context_params params;
         params.use_gpu = false;
         auto w_ctx = whisper_init_from_file_with_params(whisper_model_path.c_str(), params);
         if (w_ctx)
         {
            if (whisper_ctx_init_openvino_encoder(w_ctx, nullptr, device_name.c_str(), cache_path.c_str()))
            {
               wxLogError("whisper_ctx_init_openvino_encoder failed for device = %s", device_name.c_str());
               std::cout << "whisper_ctx_init_openvino_encoder failed." << std::endl;
               whisper_free(w_ctx);
               w_ctx = nullptr;
            }
         }
         else
         {
            wxLogError("whisper_init_from_file_with_params(%s, ...) failed", whisper_model_path.c_str());
         }
         return w_ctx;
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

      ctx = init_whisper_fut.get();
   }

   if (!ctx)
   {
      throw std::runtime_error("whisper.cpp context creation / initialization failed");
      std::cout << "error in whisper context initialization!" << std::endl;
      return false;
   }

   mIsCancelled = TotalProgress(0.01, TranslatableString{ wxString("Running Whisper Transcription using OpenVINO"), {} });

   if (mIsCancelled)
   {
      whisper_free(ctx);
      return false;
   }

   if (!whisper_is_multilingual(ctx)) {
      if (params.language != "en" || params.translate) {
         params.language = "en";
         params.translate = false;
         fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
      }
   }

   if (params.detect_language) {
      params.language = "auto";
   }

   fprintf(stderr, "%s: processing (%d samples, %.1f sec), %d threads, %d processors, lang = %s, task = %s, timestamps = %d ...\n",
      __func__, int(mono_samples.size()), float(mono_samples.size()) / WHISPER_SAMPLE_RATE,
      params.n_threads, params.n_processors,
      params.language.c_str(),
      params.translate ? "translate" : "transcribe",
      params.no_timestamps ? 0 : 1);

   bool ret = true;
   {
      whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

      wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

      wparams.print_realtime = false;
      wparams.print_progress = params.print_progress;
      wparams.print_timestamps = !params.no_timestamps;
      wparams.print_special = params.print_special;
      wparams.translate = params.translate;
      wparams.language = params.language.c_str();
      wparams.detect_language = params.detect_language;
      wparams.n_threads = params.n_threads;
      wparams.n_max_text_ctx = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
      wparams.offset_ms = params.offset_t_ms;
      wparams.duration_ms = params.duration_ms;

      wparams.token_timestamps = params.output_wts || params.max_len > 0;
      wparams.thold_pt = params.word_thold;
      wparams.max_len = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
      wparams.split_on_word = params.split_on_word;

      wparams.speed_up = params.speed_up;

      wparams.initial_prompt = params.prompt.c_str();

      wparams.greedy.best_of = params.best_of;
      wparams.beam_search.beam_size = params.beam_size;

      wparams.temperature_inc = params.no_fallback ? 0.0f : wparams.temperature_inc;
      wparams.entropy_thold = params.entropy_thold;
      wparams.logprob_thold = params.logprob_thold;

      //wparams.openvino_encode_device = params.openvino_encode_device.c_str();

      std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM
      whisper_print_user_data user_data = { &params, &pcmf32s };

      // this callback is called on each new segment
      wparams.new_segment_callback = whisper_print_segment_callback;
      wparams.new_segment_callback_user_data = &user_data;

      wparams.progress_callback = whisper_print_progress_callback;
      wparams.progress_callback_user_data = this;

      wparams.encoder_begin_callback = whisper_encoder_callback;
      wparams.encoder_begin_callback_user_data = this;

      if (whisper_full_parallel(ctx, wparams, mono_samples.data(), mono_samples.size(), params.n_processors) != 0) {
         whisper_free(ctx);
         throw std::runtime_error("whisper_full_parallel failed.");
      }
   }

   whisper_free(ctx);
   return ret;
}

std::unique_ptr<EffectEditor> EffectOVWhisperTranscription::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess& access,
   const EffectOutputs*)
{
   S.AddSpace(0, 5);
   S.StartVerticalLay();
   {
      S.StartMultiColumn(4, wxCENTER);
      {
         //m_deviceSelectionChoice
         mTypeChoiceDeviceCtrl = S.Id(ID_Type)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_deviceSelectionChoice)
            .AddChoice(XXO("OpenVINO Inference Device:"),
               Msgids(mGuiDeviceSelections.data(), mGuiDeviceSelections.size()));
      }
      S.EndMultiColumn();

      S.StartMultiColumn(4, wxCENTER);
      {
         //m_deviceSelectionChoice
         mTypeChoiceModelCtrl = S.Id(ID_Type_Model)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_modelSelectionChoice)
            .AddChoice(XXO("Whisper Model:"),
               Msgids(mGuiModelSelections.data(), mGuiModelSelections.size()));
      }
      S.EndMultiColumn();

      S.StartMultiColumn(4, wxCENTER);
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

   }
   S.EndVerticalLay();

   return nullptr;
}

