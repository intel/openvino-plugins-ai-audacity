// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include <openvino/openvino.hpp>
#include "OVMusicGenerationV2.h"
#include "WaveTrack.h"
#include "EffectOutputTracks.h"
#include "effects/EffectEditor.h"
#include <math.h>
#include <iostream>
#include <wx/log.h>

#include "BasicUI.h"
#include "ViewInfo.h"
#include "TimeWarper.h"
#include "LoadEffects.h"
#include "htdemucs.h"
#include "widgets/NumericTextCtrl.h"

#include <wx/intl.h>
#include <wx/valgen.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/checkbox.h>
#include <wx/wrapsizer.h>

#include "ShuttleGui.h"

#include "widgets/valnum.h"
#include <wx/choice.h>
#include "FileNames.h"
#include "CodeConversions.h"
#include "SyncLock.h"

#include <future>

#include "InterpolateAudio.h"

const ComponentInterfaceSymbol EffectOVMusicGenerationV2::Symbol{ XO("OpenVINO Music Generation V2") };

namespace { BuiltinEffectsModule::Registration< EffectOVMusicGenerationV2 > reg; }

BEGIN_EVENT_TABLE(EffectOVMusicGenerationV2, wxEvtHandler)
   EVT_BUTTON(ID_Type_UnloadModelsButton, EffectOVMusicGenerationV2::OnUnloadModelsButtonClicked)
END_EVENT_TABLE()

EffectOVMusicGenerationV2::EffectOVMusicGenerationV2()
{
   
   Parameters().Reset(*this);

   ov::Core core;

   //Find all supported devices on this system
   std::vector<std::string> devices = core.get_available_devices();

   for (auto d : devices)
   {
      //GNA devices are not supported
      if (d.find("GNA") != std::string::npos) continue;

      mGuiDeviceVPUSupportedSelections.push_back(wxString(d));

      if (d == "NPU") continue;

      mGuiDeviceNonVPUSupportedSelections.push_back(wxString(d));
   }

   std::vector<std::string> context_length_choices = { "5 Seconds", "10 Seconds" };
   for (auto d : context_length_choices)
   {
      mGuiContextLengthSelections.push_back({ TranslatableString{ wxString(d), {}} });
   }

   std::vector<std::string> model_selection_choices = { "musicgen-small-fp16",
                                                        "musicgen-small-int8" };
   for (auto d : model_selection_choices)
   {
      mGuiModelSelections.push_back({ TranslatableString{ wxString(d), {}} });
   }

}

EffectOVMusicGenerationV2::~EffectOVMusicGenerationV2()
{
   //cpp_stable_diffusion_ov::ModelCollateralCache::instance()->Reset();
   _musicgen = {};
}


// ComponentInterface implementation

ComponentInterfaceSymbol EffectOVMusicGenerationV2::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVMusicGenerationV2::GetDescription() const
{
   return XO("Generates an audio track from a set of text prompts");
}

VendorSymbol EffectOVMusicGenerationV2::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

// EffectDefinitionInterface implementation

EffectType EffectOVMusicGenerationV2::GetType() const
{
   return EffectTypeGenerate;
}

static void NormalizeSamples(std::shared_ptr<std::vector<float>> samples, WaveTrack* base, float target_rms)
{
   auto tmp_tracklist = base->WideEmptyCopy();

   auto iter =
      (*tmp_tracklist->Any<WaveTrack>().begin())->Channels().begin();

   auto& tmp = **iter++;
   tmp.Append((samplePtr)samples->data(), floatSample, samples->size());

   auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();
   pTmpTrack->Flush();

   float tmp_rms = pTmpTrack->GetRMS(pTmpTrack->GetStartTime(), pTmpTrack->GetEndTime());

   float rms_ratio = target_rms / tmp_rms;
   {
      float* pSamples = samples->data();
      for (size_t si = 0; si < samples->size(); si++)
      {
         pSamples[si] *= rms_ratio;
      }
   }
}

bool EffectOVMusicGenerationV2::MusicGenCallback(float perc_complete)
{
   std::lock_guard<std::mutex> guard(mProgMutex);

   mProgressFrac = perc_complete;

   if (mIsCancelled)
   {
      std::cout << "Triggering cancel." << std::endl;
      return false;
   }
}

static bool musicgen_callback(float perc_complete, void* user)
{
   EffectOVMusicGenerationV2* music_gen = (EffectOVMusicGenerationV2*)user;

   if (music_gen)
   {
      return music_gen->MusicGenCallback(perc_complete);
   }

   return true;
}

// Effect implementation
bool EffectOVMusicGenerationV2::Process(EffectInstance&, EffectSettings& settings)
{
   if (!mDurationT || (mDurationT->GetValue() <= 0))
   {
      std::cout << "Duration <= 0... returning" << std::endl;
      return false;
   }

   mIsCancelled = false;

   EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
   bool bGoodResult = true;

   {
      FilePath model_folder = FileNames::MkDir(wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath());
      std::string musicgen_model_folder = audacity::ToUTF8(wxFileName(model_folder, wxString("musicgen"))
         .GetFullPath());

      std::cout << "musicgen_model_folder = " << musicgen_model_folder << std::endl;

      
      auto text_encoder_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_TextEncoder->GetString(m_deviceSelectionChoice_TextEncoder));
      
      auto musicgen_dec0_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_UNetPositive->GetString(m_deviceSelectionChoice_MusicGenDecode0));
      auto musicgen_dec1_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_UNetNegative->GetString(m_deviceSelectionChoice_MusicGenDecode1));

      MusicGenConfig::ContinuationContext continuation_context;
      if (m_contextLengthChoice == 0)
      {
         std::cout << "continuation context of 5 seconds..." << std::endl;
         continuation_context = MusicGenConfig::ContinuationContext::FIVE_SECONDS;
      }
      else
      {
         std::cout << "continuation context of 10 seconds..." << std::endl;
         continuation_context = MusicGenConfig::ContinuationContext::TEN_SECONDS;
      }

      MusicGenConfig::ModelSelection model_selection;
      if (m_modelSelectionChoice == 0)
      {
         model_selection = MusicGenConfig::ModelSelection::MUSICGEN_SMALL_FP16;
      }
      else
      {
         model_selection = MusicGenConfig::ModelSelection::MUSICGEN_SMALL_INT8;
      }

      std::cout << "text_encoder_device = " << text_encoder_device << std::endl;
      std::cout << "MusicGen Decode Device 0 = " << musicgen_dec0_device << std::endl;
      std::cout << "MusicGen Decode Device 1 = " << musicgen_dec1_device << std::endl;


      FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());
      std::string cache_path = audacity::ToUTF8(wxFileName(cache_folder).GetFullPath());
      std::cout << "cache path = " << cache_path << std::endl;

      wxString added_trackName;

      try
      {
         mProgress->SetMessage(TranslatableString{ wxString("Creating MusicGen Pipeline"), {} });

         auto musicgen_pipeline_creation_future = std::async(std::launch::async,
            [this, &musicgen_model_folder, &cache_path, &text_encoder_device, &musicgen_dec0_device, &musicgen_dec1_device,
             &continuation_context, &model_selection]
            {

               //TODO: This should be much more efficient. No need to destroy *everything*, just update the
               // pieces of the pipelines that have changed.
               if ((musicgen_dec0_device != _musicgen_config.musicgen_decode_device0) ||
                  (musicgen_dec1_device != _musicgen_config.musicgen_decode_device1))
               {
                  //force destroy music gen if config has changed.
                  _musicgen = {};
               }

               if (continuation_context != _musicgen_config.m_continuation_context)
               {
                  _musicgen = {};
               }

               if (model_selection != _musicgen_config.model_selection)
               {
                  _musicgen = {};
               }


               if (!_musicgen)
               {
                  _musicgen_config.musicgen_decode_device0 = musicgen_dec0_device;
                  _musicgen_config.musicgen_decode_device1 = musicgen_dec1_device;

                  _musicgen_config.cache_folder = cache_path;
                  _musicgen_config.model_folder = musicgen_model_folder;

                  _musicgen_config.m_continuation_context = continuation_context;

                  if (_musicgen_config.musicgen_decode_device0 == "CPU")
                  {
                     _musicgen_config.initial_decode_device = "CPU";
                  }
                  else
                  {
                     _musicgen_config.initial_decode_device = "GPU";
                  }

                  _musicgen_config.model_selection = model_selection;

                  _musicgen = std::make_shared< MusicGen >(_musicgen_config);
               }
            });

         float total_time = 0.f;
         std::future_status status;
         do {
            using namespace std::chrono_literals;
            status = musicgen_pipeline_creation_future.wait_for(0.5s);

            {
               std::string message = "Loading Music Generation AI Models... ";
               if (total_time > 30)
               {
                  message += " (This could take a while if this is the first time running this feature with this device)";
               }
               if (TotalProgress(0.01, TranslatableString{ wxString(message), {} }))
               {
                  mIsCancelled = true;
               }
            }
         } while (status != std::future_status::ready);

         if (mIsCancelled)
         {
            return false;
         }

         if (!_musicgen)
         {
            wxLogError("MusicGen pipeline could not be created.");
            return false;
         }

         _prompt = audacity::ToUTF8(mTextPrompt->GetLineText(0));

         std::string seed_str = audacity::ToUTF8(mSeed->GetLineText(0));
         _seed_str = seed_str;

         std::optional<unsigned int> seed;
         if (!seed_str.empty() && seed_str != "")
         {
            seed = std::stoul(seed_str);
         }
         else
         {
            //seed is not set.. set it to time.
            time_t t;
            seed = (unsigned)time(&t);
         }

         std::cout << "Guidance Scale = " << mGuidanceScale << std::endl;
         std::cout << "TopK = " << mTopK << std::endl;

         std::string descriptor_str = "prompt: " + _prompt;
         descriptor_str += ", seed: " + std::to_string(*seed);
         descriptor_str += ", Guidance Scale = " + std::to_string(mGuidanceScale);
         descriptor_str += ", TopK = " + std::to_string(mTopK);

         

         added_trackName = wxString("Generated: (" + descriptor_str + ")");

        

         std::cout << "Duration = " << (float)mDurationT->GetValue() << std::endl;

         std::optional< std::shared_ptr<std::vector<float>> > audio_to_continue;

         if (_AudioContinuationCheckBox->GetValue())
         {
            auto track = *(outputs.Get().Selected<WaveTrack>().begin());

            auto mono = track->GetChannel(0);

            // create a temporary track list to append samples to
            auto tmp_tracklist = track->WideEmptyCopy();

            auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();

            auto iter =
               pTmpTrack->Channels().begin();

            auto end = mT0;

            if (end > mono->GetStartTime())
            {
               auto start = mT1 - 10.f;

               if (start < mono->GetStartTime())
                  start = mono->GetStartTime();

               auto start_s = mono->TimeToLongSamples(start);
               auto end_s = mono->TimeToLongSamples(end);

               size_t total_samples = (end_s - start_s).as_size_t();

               Floats entire_input{ total_samples };

               bool bOkay = mono->GetFloats(entire_input.get(), start_s, total_samples);
               if (!bOkay)
               {
                  throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
                     std::to_string(total_samples) + "samples");
               }

               auto& tmpMono = **iter++;
               tmpMono.Append((samplePtr)entire_input.get(), floatSample, total_samples);

               //flush it
               auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();
               pTmpTrack->Flush();

               if (pTmpTrack->GetRate() != 32000)
               {
                  pTmpTrack->Resample(32000, mProgress);
               }

               {
                  auto pResampledTrack = pTmpTrack->GetChannel(0);

                  start = pResampledTrack->GetStartTime();
                  end = pResampledTrack->GetEndTime();

                  auto start_s = pResampledTrack->TimeToLongSamples(start);
                  auto end_s = pResampledTrack->TimeToLongSamples(end);

                  total_samples = (end_s - start_s).as_size_t();

                  std::shared_ptr< std::vector<float> > resampled_samples = std::make_shared< std::vector<float> >(total_samples);
                  bool bOkay = pResampledTrack->GetFloats(resampled_samples->data(), start_s, total_samples);
                  if (!bOkay)
                  {
                     throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
                        std::to_string(total_samples) + "samples");
                  }

                  audio_to_continue = resampled_samples;

                  std::cout << "okay, set audio to continue to " << total_samples << " samples..." << std::endl;
               }
            }
         }

         auto musicgen_pipeline_run_future = std::async(std::launch::async,
            [this, &seed, &audio_to_continue]
            {

               CallbackParams callback_params;
               callback_params.user = this;
               callback_params.every_n_new_tokens = 5;
               callback_params.callback = musicgen_callback;

               auto wav = _musicgen->Generate(_prompt,
                  audio_to_continue, 
                  (float)mDurationT->GetValue(),
                  seed,
                  mGuidanceScale,
                  mTopK,
                  callback_params);

               return wav;
            }
            );

         mProgressFrac = 0.0;
         mProgMessage = "Running Music Generation";

         do {
            using namespace std::chrono_literals;
            status = musicgen_pipeline_run_future.wait_for(0.5s);
            {
               std::lock_guard<std::mutex> guard(mProgMutex);
               mProgress->SetMessage(TranslatableString{ wxString(mProgMessage), {} });
               if (TotalProgress(mProgressFrac))
               {
                  mIsCancelled = true;
               }
            }

         } while (status != std::future_status::ready);

         std::cout << "generated!" << std::endl;

         auto generated_samples_L = musicgen_pipeline_run_future.get();
         if (!generated_samples_L)
            return false;

         std::shared_ptr<std::vector<float>> generated_samples_R;

         settings.extra.SetDuration(mDurationT->GetValue());
         const auto duration = settings.extra.GetDuration();

         //clip samples to max duration
         size_t max_output_samples = duration * 32000;
         if (generated_samples_L)
         {
            std::cout << "Clipping Left from " << generated_samples_L->size() << " to " << max_output_samples << " samples." << std::endl;
            if (generated_samples_L->size() > max_output_samples)
            {
               generated_samples_L->resize(max_output_samples);
            }
         }
         else
         {
            std::cout << "No L samples" << std::endl;
            return false;
         }

         if (generated_samples_R)
         {
            std::cout << "Clipping Right from " << generated_samples_R->size() << " to " << max_output_samples << " samples." << std::endl;
            if (generated_samples_R->size() > max_output_samples)
            {
               generated_samples_R->resize(max_output_samples);
            }
         }

         bool bNormalized = false;
         for (auto track : outputs.Get().Selected<WaveTrack>())
         {
            bool editClipCanMove = GetEditClipsCanMove();

//Don't normalize until we figure out how we want this to work
// (as you need to keep in mind audio continuation, etc.).
#if 0
            if (!bNormalized)
            {
#define GEN_DB_TO_LINEAR(x) (pow(10.0, (x) / 20.0))
               float target_rms = GEN_DB_TO_LINEAR(std::clamp<double>(mRMSLevel, -145.0, 0.0));
               NormalizeSamples(generated_samples_L, track, target_rms);
               if (generated_samples_R)
               {
                  NormalizeSamples(generated_samples_R, track, target_rms);
               }
               bNormalized = true;
            }
#endif

            //if we can't move clips, and we're generating into an empty space,
            //make sure there's room.
            if (!editClipCanMove &&
               track->IsEmpty(mT0, mT1 + 1.0 / track->GetRate()) &&
               !track->IsEmpty(mT0,
                  mT0 + duration - (mT1 - mT0) - 1.0 / track->GetRate()))
            {
               EffectUIServices::DoMessageBox(*this,
                  XO("There is not enough room available to generate the audio"),
                  wxICON_STOP,
                  XO("Error"));
               return false;
            }

            if (duration > 0.0)
            {
               auto pProject = FindProject();

               // Create a temporary track
               track->SetName(added_trackName);

               // create a temporary track list to append samples to
               auto tmp_tracklist = track->WideEmptyCopy();
               auto iter =
                  (*tmp_tracklist->Any<WaveTrack>().begin())->Channels().begin();

               //append output samples to L & R channels.
               auto& tmpLeft = **iter++;
               tmpLeft.Append((samplePtr)generated_samples_L->data(), floatSample, generated_samples_L->size());

               if (track->NChannels() > 1)
               {
                  auto& tmpRight = **iter;

                  if (generated_samples_R)
                  {
                     std::cout << "appending right samples" << std::endl;
                     tmpRight.Append((samplePtr)generated_samples_R->data(), floatSample, generated_samples_R->size());
                  }
                  else
                  {
                     std::cout << "appending right samples from left" << std::endl;
                     tmpRight.Append((samplePtr)generated_samples_L->data(), floatSample, generated_samples_L->size());
                  }
               }

               //flush it
               auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();
               pTmpTrack->Flush();

               // The track we just populated with samples is 32000 Hz.
               // Within the ClearAndPaste() operation below, it will automatically
               // resample this to whatever the 'target' track sample rate is.
               pTmpTrack->SetRate(32000);

               PasteTimeWarper warper{ mT1, mT0 + duration };
               const auto& selectedRegion =
                  ViewInfo::Get(*pProject).selectedRegion;

               track->ClearAndPaste(
                  selectedRegion.t0(), selectedRegion.t1(),
                  *tmp_tracklist, true, false, &warper);


               if (!bGoodResult) {
                  return false;
               }
            }
            else
            {
               // If the duration is zero, there's no need to actually
               // generate anything
               track->Clear(mT0, mT1);
            }
         }

         if (mIsCancelled)
         {
            return false;
         }

         if (bGoodResult) {

            outputs.Commit();

            mT1 = mT0 + duration; // Update selection.
         }

      }
      catch (const std::exception& error) {
         wxLogError("In Music Generation V2, exception: %s", error.what());
         EffectUIServices::DoMessageBox(*this,
            XO("Music Generation failed. See details in Help->Diagnostics->Show Log..."),
            wxICON_STOP,
            XO("Error"));
         return false;
      }

      std::cout << "returning!" << std::endl;
      return bGoodResult;
   }
}

bool EffectOVMusicGenerationV2::UpdateProgress(double perc)
{
   if (!TotalProgress(perc / 100.0))
   {
      std::cout << "Total Progress returned false" << std::endl;
      return false;
   }

   return true;
}

std::unique_ptr<EffectEditor> EffectOVMusicGenerationV2::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess& access,
   const EffectOutputs*)
{
   DoPopulateOrExchange(S, access);
   return nullptr;
}


void EffectOVMusicGenerationV2::DoPopulateOrExchange(
   ShuttleGui& S, EffectSettingsAccess& access)
{
   mUIParent = S.GetParent();

   //EnablePreview(false); //Port this

   
   
   S.StartVerticalLay(wxLEFT);
   {
      S.StartMultiColumn(3, wxLEFT);
      {
         mUnloadModelsButton = S.Id(ID_Type_UnloadModelsButton).AddButton(XO("Unload Models"));

         if (!_musicgen)
         {
            mUnloadModelsButton->Enable(false);
         }
      }
      S.EndMultiColumn();

// Disable Normalization option until we figure out how we want this to work..
#if 0
      S.StartMultiColumn(4, wxLEFT);
      {
         S.AddVariableText(XO("Normalize "), false,
            wxALIGN_CENTER_VERTICAL | wxALIGN_LEFT);

         S.Name(XO("RMS dB"))
            .Validator<FloatingPointValidator<double>>(
               2, &mRMSLevel,
               NumValidatorStyle::ONE_TRAILING_ZERO,
               -145.0, 0.0)
            .AddTextBox({}, L"", 10);

         S.AddVariableText(XO("dB"), false,
            wxALIGN_CENTER_VERTICAL | wxALIGN_LEFT);
      }
      S.EndMultiColumn();
#endif


      S.StartMultiColumn(2, wxLEFT);
      {
         S.AddPrompt(XXO("&Duration:"));
         auto& extra = access.Get().extra;
         mDurationT = safenew
            NumericTextCtrl(FormatterContext::SampleRateContext(mProjectRate),
               S.GetParent(), wxID_ANY,
               NumericConverterType_TIME(),
               extra.GetDurationFormat(),
               extra.GetDuration(),
               NumericTextCtrl::Options{}
         .AutoPos(true));
         S.Id(ID_Type_Duration).Name(XO("Duration"))
            .Position(wxALIGN_LEFT | wxALL)
            .AddWindow(mDurationT);

      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         mTypeChoiceModelSelection = S.Id(ID_Type_ModelSelection)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_modelSelectionChoice)
            .AddChoice(XXO("Model Selection:"),
               Msgids(mGuiModelSelections.data(), mGuiModelSelections.size()));

        
      }
      S.EndMultiColumn();

      S.StartMultiColumn(4, wxCENTER);
      {
            S.StartMultiColumn(2, wxCENTER);
            {
               mTextPrompt = S.Id(ID_Type_Prompt)
                  .Style(wxTE_LEFT)
                  .AddTextBox(XXO("Prompt:"), wxString(_prompt), 60);
            }
            S.EndMultiColumn();

            S.StartMultiColumn(2, wxCENTER);
            {

               mTypeChoiceDeviceCtrl_TextEncoder = S.Id(ID_Type_TxtEncoder)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_deviceSelectionChoice_TextEncoder)
                  .AddChoice(XXO("Text Encoder Device:"),
                     Msgids(mGuiDeviceNonVPUSupportedSelections.data(), mGuiDeviceNonVPUSupportedSelections.size()));

               mTypeChoiceDeviceCtrl_UNetPositive = S.Id(ID_Type_MusicGenDecodeDevice0)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_deviceSelectionChoice_MusicGenDecode0)
                  .AddChoice(XXO("MusicGen Decode Device:"),
                     Msgids(mGuiDeviceVPUSupportedSelections.data(), mGuiDeviceVPUSupportedSelections.size()));

               mTypeChoiceDeviceCtrl_UNetNegative = S.Id(ID_Type_MusicGenDecodeDevice1)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_deviceSelectionChoice_MusicGenDecode1)
                  .AddChoice(XXO("MusicGen Decode Device:"),
                     Msgids(mGuiDeviceVPUSupportedSelections.data(), mGuiDeviceVPUSupportedSelections.size()));

               //mTypeChoiceScheduler->Hide();
            }
            S.EndMultiColumn();
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         mSeed = S.Id(ID_Type_Seed)
            .Style(wxTE_LEFT)
            .AddNumericTextBox(XXO("Seed:"), wxString(_seed_str), 10);

         auto t0 = S.Name(XO("Guidance Scale"))
            .Validator<FloatingPointValidator<float>>(
               6, &mGuidanceScale,
               NumValidatorStyle::NO_TRAILING_ZEROES,
               0.0f,
               10.0f)
            .AddTextBox(XO("Guidance Scale"), L"", 12);

         mTopKCtl = S.Id(ID_Type_TopK)
            .Validator<IntegerValidator<int>>(&mTopK,
               NumValidatorStyle::DEFAULT,
               10,
               1000)
            .AddTextBox(XO("TopK"), L"", 12);
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         mTypeChoiceContextLength = S.Id(ID_Type_ContextLength)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_contextLengthChoice)
            .AddChoice(XXO("Context Length:"),
               Msgids(mGuiContextLengthSelections.data(), mGuiContextLengthSelections.size()));
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         _AudioContinuationCheckBox = S.AddCheckBox(XXO("&Audio Continuation"), false);

         EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
         auto track_selection_size = outputs.Get().Selected<WaveTrack>().size();
         std::cout << "Track Selection Size = " << track_selection_size << std::endl;

         if (track_selection_size != 1)
         {
            _AudioContinuationCheckBox->Enable(false);
         }
         else
         {
            auto track = *(outputs.Get().Selected<WaveTrack>().begin());

            auto t0 = track->GetStartTime();
            auto t1 = track->GetEndTime();

            std::cout << "track end time = " << t1 << std::endl;
            std::cout << "mT0 = " << mT0 << std::endl;
            if (track->IsEmpty(t0, t1) || (track->Channels().size() > 1))
            {
               _AudioContinuationCheckBox->Enable(false);
            }
            else
            {
               //if the start position of the user's track selection is within 0.1 seconds of the end time, we can
               // make a pretty safe assumption that their intention is to perform music continuation.
               if (((mT0 - t1) >= 0.) && ((mT0 - t1) <= 0.1))
               {
                  _AudioContinuationCheckBox->SetValue(true);
               }
            }
         }

         

         
      }
      S.EndMultiColumn();

   }

   
   S.EndVerticalLay();

}


void EffectOVMusicGenerationV2::OnUnloadModelsButtonClicked(wxCommandEvent& evt)
{

   std::cout << "Unload clicked, but it's not hooked up to anything (yet)!" << std::endl;
#if 0
   if (mUnloadModelsButton)
   {
      mUnloadModelsButton->Enable(false);
   }
#endif
}



bool EffectOVMusicGenerationV2::TransferDataToWindow(const EffectSettings &settings)
{
   if (!mUIParent->TransferDataToWindow())
   {
      return false;
   }

   EffectEditor::EnablePreview(mUIParent, false);

   std::cout << "settings.extra.GetDuration() = " << settings.extra.GetDuration() << std::endl;
   mDurationT->SetValue(settings.extra.GetDuration());
   return true;

}

bool EffectOVMusicGenerationV2::TransferDataFromWindow(EffectSettings & settings)
{

   if (!mUIParent->Validate() || !mUIParent->TransferDataFromWindow())
   {
      return false;
   }


   settings.extra.SetDuration(mDurationT->GetValue());
   //EnablePreview(false);  //Port this
   return true;
}
