// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include <openvino/openvino.hpp>
#include "OVMusicGeneration.h"
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
#include <wx/wrapsizer.h>

#include "ShuttleGui.h"

#include "widgets/valnum.h"
#include <wx/choice.h>
#include "FileNames.h"
#include "CodeConversions.h"
#include "SyncLock.h"

#include "cpp_stable_diffusion_audio_ov/stable_diffusion_audio_interpolation_pipeline.h"
#include "cpp_stable_diffusion_ov/stable_diffusion_pipeline.h"
#include "cpp_stable_diffusion_ov/model_collateral_cache.h"
#include "cpp_stable_diffusion_audio_ov/spectrogram_image_converter.h"
#include <future>

#include "InterpolateAudio.h"

const ComponentInterfaceSymbol EffectOVMusicGeneration::Symbol{ XO("OpenVINO Music Generation") };

namespace { BuiltinEffectsModule::Registration< EffectOVMusicGeneration > reg; }

BEGIN_EVENT_TABLE(EffectOVMusicGeneration, wxEvtHandler)
   EVT_CHOICE(ID_Type_Mode, EffectOVMusicGeneration::OnChoice)
   EVT_CHOICE(ID_Type_SeedImageSimple, EffectOVMusicGeneration::OnSimpleSeedImageChoice)
   //EVT_TIMETEXTCTRL_UPDATED(ID_Type_Duration, EffectOVMusicGeneration::OnSimpleSeedImageChoice)
   EVT_COMMAND(ID_Type_Duration, wxEVT_COMMAND_TEXT_UPDATED, EffectOVMusicGeneration::OnSimpleSeedImageChoice)
   EVT_TEXT(ID_Type_NumInterpolationSteps, EffectOVMusicGeneration::OnSimpleSeedImageChoice)
   EVT_BUTTON(ID_Type_UnloadModelsButton, EffectOVMusicGeneration::OnUnloadModelsButtonClicked)
END_EVENT_TABLE()

EffectOVMusicGeneration::EffectOVMusicGeneration()
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

      if (d == "NPU")
      {
         m_deviceSelectionChoice_UNetNegative = mGuiDeviceVPUSupportedSelections.size() - 1;
      }

      if (d == "NPU") continue;

      mGuiDeviceNonVPUSupportedSelections.push_back(wxString(d));

      if (d == "GPU")
      {
         m_deviceSelectionChoice_VAEEncoder = mGuiDeviceNonVPUSupportedSelections.size() - 1;
         m_deviceSelectionChoice_VAEDecoder = mGuiDeviceNonVPUSupportedSelections.size() - 1;
         m_deviceSelectionChoice_UNetPositive = mGuiDeviceNonVPUSupportedSelections.size() - 1;
      }
   }

   std::vector<std::string> seedImages = { "og_beat", "agile", "marim", "motorway", "vibes" };
   for (auto d : seedImages)
   {
      mGuiSeedImageSelections.push_back({ TranslatableString{ wxString(d), {}} });
      mGuiSeedImageSelectionsSimple.push_back({ TranslatableString{ wxString(d), {}} });
   }

   mGuiSeedImageSelectionsSimple.push_back({ TranslatableString{ wxString("none"), {}}});

   std::vector<std::string> schedulerChoices = { "EulerDiscreteScheduler", "USTMScheduler", "PNDMScheduler" };
   for (auto d : schedulerChoices)
   {
      mGuiSchedulerSelections.push_back({ TranslatableString{ wxString(d), {}} });
   }

   std::vector<std::string> modeChoices = { "Simple", "Advanced" };
   for (auto d : modeChoices)
   {
      mGuiModeSelections.push_back({ TranslatableString{ wxString(d), {}} });
   }
}

EffectOVMusicGeneration::~EffectOVMusicGeneration()
{
   cpp_stable_diffusion_ov::ModelCollateralCache::instance()->Reset();
}


// ComponentInterface implementation

ComponentInterfaceSymbol EffectOVMusicGeneration::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVMusicGeneration::GetDescription() const
{
   return XO("Generates an audio track from a set of text prompts");
}

VendorSymbol EffectOVMusicGeneration::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

// EffectDefinitionInterface implementation

EffectType EffectOVMusicGeneration::GetType() const
{
   return EffectTypeGenerate;
}

bool EffectOVMusicGeneration::interp_callback(size_t interp_step_i_complete)
{
   mInterp_steps_complete = interp_step_i_complete + 1;

   return true;
}

bool EffectOVMusicGeneration::unet_callback_for_interp(size_t unet_step_i_complete, size_t unet_total_iterations)
{
   std::lock_guard<std::mutex> guard(mProgMutex);

   mProgMessage = "Audio Segment " + std::to_string(mInterp_steps_complete + 1) + " / " + std::to_string(mNumOutputSegments) +
      ": Running UNet iteration " + std::to_string(unet_step_i_complete + 1) + " / " + std::to_string(unet_total_iterations);

   mNumUnetIterationsComplete++;
   if (mNumUnetIterationsComplete == mNumTotalUnetIterations)
   {
      mProgMessage = " Finalizing Waveform...";
   }

   if (mNumTotalUnetIterations)
   {
      mProgressFrac = (float)mNumUnetIterationsComplete / (float)mNumTotalUnetIterations;
   }

   if (mIsCancelled)
   {
      std::cout << "Triggering cancel." << std::endl;
      return false;
   }

   return true;
}

static bool MyInterpolationCallback(size_t interp_step_i_complete,
   size_t num_interp_steps,
   std::shared_ptr<std::vector<float>> wav,
   std::shared_ptr<std::vector<uint8_t>> img_rgb,
   size_t img_width,
   size_t img_height,
   void* user)
{
   std::cout << "Interpolation iteration " << interp_step_i_complete << " / " << num_interp_steps << " complete." << std::endl;
   EffectOVMusicGeneration* r = (EffectOVMusicGeneration*)user;
   return r->interp_callback(interp_step_i_complete);
}

static bool MyUnetForInterpCallback(size_t unet_step_i_complete,
   size_t num_unet_steps,
   void* user)
{
   std::cout << "unet iteration " << unet_step_i_complete << " / " << num_unet_steps << " complete." << std::endl;
   EffectOVMusicGeneration* r = (EffectOVMusicGeneration*)user;
   return r->unet_callback_for_interp(unet_step_i_complete, num_unet_steps);
}

bool EffectOVMusicGeneration::unet_callback(size_t unet_step_i_complete, size_t unet_total_iterations)
{
   std::lock_guard<std::mutex> guard(mProgMutex);

   mProgMessage = "Stable Diffusion: Running UNet iteration " + std::to_string(unet_step_i_complete + 1) + " / " + std::to_string(unet_total_iterations);

   size_t total_iterations = unet_total_iterations;
   size_t iterations_complete = (unet_step_i_complete + 1);

   if (iterations_complete == total_iterations)
   {
      mProgMessage = " Converting Spectrogram to Waveform...";
   }

   if (total_iterations)
   {
      mProgressFrac = (float)iterations_complete / (float)total_iterations;
   }

   if (mIsCancelled)
   {
      std::cout << "Triggering cancel." << std::endl;
      return false;
   }

   return true;
}

static bool MyUnetCallback(size_t unet_step_i_complete,
   size_t num_unet_steps,
   void* user)
{
   std::cout << "unet iteration " << unet_step_i_complete << " / " << num_unet_steps << " complete." << std::endl;
   EffectOVMusicGeneration* r = (EffectOVMusicGeneration*)user;
   return r->unet_callback(unet_step_i_complete, num_unet_steps);
}

static std::shared_ptr< std::vector<float> > spec_to_wav(std::shared_ptr< std::vector<uint8_t> > spec, int channel)
{
   try
   {
      cpp_stable_diffusion_ov::SpectrogramImageConverter converter;
      return converter.audio_from_spectrogram_image(spec, 512, 512, channel);
   }
   catch (const std::exception& error) {
      std::cout << "converter.audio_from_spectrogram_image: " << error.what() << std::endl;
   }

   return {};
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

//smooth the segment seams to create a more seamless transition between
// the merged segments (i.e. remove clicks / pops in the audio)
static void repair_audio_seams(std::shared_ptr<std::vector<float>> generated_samples, int num_output_segments)
{
   if (generated_samples)
   {
      const size_t num_samples_per_step = 225351;

      const size_t len = 256;
      const size_t repair_len = 32;

      for (size_t i = 1; i < (num_output_segments - 1); i++)
      {
         size_t start = i * num_samples_per_step - (len/2);
         if ((start + len) < generated_samples->size())
         {
            float* pStart = generated_samples->data() + start;
            size_t repair_start = (len/2) - (repair_len/2);
            InterpolateAudio(pStart, len, repair_start, repair_len);
         }
      }
   }
}

#define SAMPLES_PER_GEN_SEGMENT (225351.0)
#define SECONDS_PER_GEN_SEGMENT (SAMPLES_PER_GEN_SEGMENT / 44100.0)
#define DEFAULT_SIMPLE_NUM_INTERPOLATION_STEPS 5

// Effect implementation
bool EffectOVMusicGeneration::Process(EffectInstance&, EffectSettings& settings)
{
   if (!mDurationT || (mDurationT->GetValue() <= 0))
   {
      std::cout << "Duration <= 0... returning" << std::endl;
      return false;
   }

   mIsCancelled = false;

   mInterp_steps_complete = 0;

   EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
   bool bGoodResult = true;

   {
      FilePath model_folder = FileNames::MkDir(wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath());
      std::string riffusion_model_folder = audacity::ToUTF8(wxFileName(model_folder, wxString("riffusion-unet-quantized-int8"))
         .GetFullPath());

      std::cout << "riffusion_model_folder = " << riffusion_model_folder << std::endl;

      
      auto text_encoder_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_TextEncoder->GetString(m_deviceSelectionChoice_TextEncoder));
      auto vae_encoder_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_VAEEncoder->GetString(m_deviceSelectionChoice_VAEEncoder));
      auto unet_pos_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_UNetPositive->GetString(m_deviceSelectionChoice_UNetPositive));
      auto unet_neg_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_UNetNegative->GetString(m_deviceSelectionChoice_UNetNegative));
      auto vae_decoder_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_VAEDecoder->GetString(m_deviceSelectionChoice_VAEDecoder));

      std::cout << "text_encoder_device = " << text_encoder_device << std::endl;
      std::cout << "vae_encoder_device = " << vae_encoder_device << std::endl;
      std::cout << "unet_pos_device = " << unet_pos_device << std::endl;
      std::cout << "unet_neg_device = " << unet_neg_device << std::endl;
      std::cout << "vae_decoder_device = " << vae_decoder_device << std::endl;

      auto scheduler = audacity::ToUTF8(mTypeChoiceScheduler->GetString(m_schedulerSelectionChoice));
      std::cout << "scheduler = " << scheduler << std::endl;

      bool bAdvanced = (m_modeSelectionChoice == 1);

      FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());
      std::string cache_path = audacity::ToUTF8(wxFileName(cache_folder).GetFullPath());
      std::cout << "cache path = " << cache_path << std::endl;

      wxString added_trackName;

      // If any selected tracks have channels > 1, we go into stereo mode.
      bool bStereo = false;
      for (auto pOutWaveTrack : outputs.Get().Selected<WaveTrack>())
      {
         if (pOutWaveTrack->NChannels() > 1)
         {
            bStereo = true;
            break;
         }
      }

      std::cout << "Generate Stereo = " << bStereo << std::endl;

      std::shared_ptr<std::vector<float>> generated_samples_L;
      std::shared_ptr<std::vector<float>> generated_samples_R;

      
      unsigned int start_seed;
      unsigned int end_seed;

      _start_seed_str = audacity::ToUTF8(mStartSeed->GetLineText(0));
      _end_seed_str = audacity::ToUTF8(mEndSeed->GetLineText(0));
      if (!_start_seed_str.empty() && _start_seed_str != "")
      {
         start_seed = std::stoul(_start_seed_str);
      }
      else
      {
         //seed is not set.. set it to time.
         time_t t;
         start_seed = (unsigned)time(&t);
      }

      if (!_end_seed_str.empty() && _end_seed_str != "")
      {
         end_seed = std::stoul(_end_seed_str);
      }
      else
      {
         //seed is not set.. set it to time + 1.
         time_t t;
         end_seed = (unsigned)time(&t) + 1;
      }
      std::string pos_prompt_start = audacity::ToUTF8(mTextPromptStart->GetLineText(0));
      std::string pos_prompt_end = audacity::ToUTF8(mTextPromptEnd->GetLineText(0));
      std::string neg_prompt_end = "";
      std::string seed_image = audacity::ToUTF8(mTypeChoiceSeedImage->GetString(m_seedImageSelectionChoice));
      
      float start_denoising = mStartDenoising;
      float end_denoising = mEndDenoising;
      float guidance_scale = mGuidanceScaleAdvanced;
      int num_inference_steps = mNumInferenceStepsAdvanced;
      int num_interpolation_steps = mNumInterpolationSteps;


      bool bSimpleMenu = !bAdvanced;
      if (!bAdvanced)
      {
         if (mTypeChoiceSeedImageSimple->GetSelection() != (mTypeChoiceSeedImageSimple->GetCount() - 1))
         {
            bAdvanced = true;

            pos_prompt_start = audacity::ToUTF8(mTextPrompt->GetLineText(0));
            pos_prompt_end = pos_prompt_start;
            neg_prompt_end = "";
            start_denoising = mDenoisingSimple; 
            end_denoising = start_denoising;
            guidance_scale = mGuidanceScale;
            num_inference_steps = mNumInferenceSteps;
            num_interpolation_steps = DEFAULT_SIMPLE_NUM_INTERPOLATION_STEPS;
            seed_image = audacity::ToUTF8(mTypeChoiceSeedImageSimple->GetString(m_seedImageSelectionChoiceSimple));

            _start_seed_str = audacity::ToUTF8(mSeed->GetLineText(0));
            if (!_start_seed_str.empty() && _start_seed_str != "")
            {
               start_seed = std::stoul(_start_seed_str);
            }
            else
            {
               //seed is not set.. set it to time.
               time_t t;
               start_seed = (unsigned)time(&t);
            }
            end_seed = start_seed + 7;
         }
      }

      if (bAdvanced)
      {
         // for each interpolation step we generate 225351 samples @ 44100 Hz, so convert that to seconds.
         //double gen_duration_in_secs = (225351.0 * num_interpolation_steps) / 44100.0;
         //settings.extra.SetDuration(gen_duration_in_secs);

         //std::cout << "duration = " << gen_duration_in_secs << std::endl;
         std::cout << "mDurationT->GetValue() = " << mDurationT->GetValue() << std::endl;

         mNumOutputSegments = (int)(std::ceil((mDurationT->GetValue() * 44100.0) / SAMPLES_PER_GEN_SEGMENT));
         std::cout << mDurationT->GetValue() << " seconds require " << mNumOutputSegments << " output segments" << std::endl;

         if (mNumOutputSegments <= 0)
         {
            std::cout << "mNumOutputSegments unexpectedly is <= 0" << std::endl;
            return false;
         }
         _pos_prompt_start = pos_prompt_start;
         _pos_prompt_end = pos_prompt_end;

         if (_pos_prompt_end.empty() || _pos_prompt_end == "")
         {
            _pos_prompt_end = _pos_prompt_start;
         }

         _neg_prompt_end = neg_prompt_end;

         std::cout << "mNumInferenceStepsAdvanced = " << num_inference_steps << std::endl;
         std::cout << "mGuidanceScaleAdvanced = " << guidance_scale << std::endl;

         std::cout << "start prompt = " << _pos_prompt_start << std::endl;
         std::cout << "mStartDenoising = " << start_denoising << std::endl;

         std::cout << "end prompt = " << _pos_prompt_end << std::endl;
         std::cout << "mEndDenoising = " << end_denoising << std::endl;

         std::cout << "negative prompt = " << _neg_prompt_end << std::endl;

         std::cout << "mNumInterpolationSteps = " << num_interpolation_steps << std::endl;
         std::cout << "seed_image = " << seed_image << std::endl;


         std::cout << "start seed = " << start_seed << std::endl;
         std::cout << "end seed = " << end_seed << std::endl;
         

         try
         {
            if (bSimpleMenu)
            {
               std::string descriptor_str = "prompt: " + _pos_prompt_start;
               descriptor_str += ", seed_image: " + seed_image;
               descriptor_str += ", seed: " + std::to_string(start_seed);
               descriptor_str += ", strength: " + std::to_string(start_denoising);
               descriptor_str += ", scale: " + std::to_string(guidance_scale);
               descriptor_str += ", steps: " + std::to_string(num_inference_steps);
               descriptor_str += ", scheduler: " + scheduler;
               added_trackName = wxString("Generated(Simple): " + descriptor_str);
            }
            else
            {
               std::string descriptor_str = "start prompt: " + _pos_prompt_start;
               descriptor_str += ", end prompt: " + _pos_prompt_end;
               if (!_neg_prompt_end.empty() && _neg_prompt_end != "")
               {
                  descriptor_str += ", negative prompt: " + _neg_prompt_end;
               }
               descriptor_str += ", seed_image: " + seed_image;
               descriptor_str += ", start seed: " + std::to_string(start_seed);
               descriptor_str += ", end seed: " + std::to_string(end_seed);
               descriptor_str += ", start strength: " + std::to_string(start_denoising);
               descriptor_str += ", end strength: " + std::to_string(end_denoising);
               descriptor_str += ", scale: " + std::to_string(guidance_scale);
               descriptor_str += ", steps: " + std::to_string(num_inference_steps);
               descriptor_str += ", scheduler: " + scheduler;
               added_trackName = wxString("Generated(Advanced): " + descriptor_str);
            }

            mProgress->SetMessage(TranslatableString{ wxString("Creating Riffusion Pipeline"), {} });
            auto riff_pipeline_creation_future = std::async(std::launch::async,
               [&riffusion_model_folder, &cache_path, &text_encoder_device, &unet_pos_device, &unet_neg_device,
               &vae_decoder_device, &vae_encoder_device] {
                  return std::make_shared< cpp_stable_diffusion_ov::StableDiffusionAudioInterpolationPipeline >(riffusion_model_folder,
                     cache_path,
                     text_encoder_device,
                     unet_pos_device,
                     unet_neg_device,
                     vae_decoder_device,
                     vae_encoder_device);
               }
            );


            float total_time = 0.f;
            std::future_status status;
            do {
               using namespace std::chrono_literals;
               status = riff_pipeline_creation_future.wait_for(0.5s);

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

            auto riffusion_pipeline = riff_pipeline_creation_future.get();

            if (mIsCancelled)
            {
               return false;
            }

            mProgress->SetMessage(TranslatableString{ wxString("Running Music Generation"), {} });

            std::pair< cpp_stable_diffusion_ov::CallbackFuncUnetIteration, void*> mycallback = { MyUnetForInterpCallback, this };
            // estimate total unet iterations so that we can accurately set % complete as the pipeline runs.
            mNumUnetIterationsComplete = 0;
            mNumTotalUnetIterations = riffusion_pipeline->EstimateTotalUnetIterations(
               start_denoising,
               end_denoising,
               num_interpolation_steps,
               num_inference_steps,
               scheduler,
               mNumOutputSegments);


            std::pair< cpp_stable_diffusion_ov::StableDiffusionAudioInterpolationPipeline::CallbackFuncInterpolationIteration, void*> myIntcallback = { MyInterpolationCallback, this };
            std::future< std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > >generated_samples_future = std::async(std::launch::async,
               [this, &bStereo, &start_denoising, &end_denoising, &guidance_scale, &num_inference_steps, &num_interpolation_steps,
               &riffusion_pipeline, &start_seed, &end_seed, &seed_image, &scheduler, &mycallback, &myIntcallback]
               {
                  return (*riffusion_pipeline)(bStereo,
                     _pos_prompt_start,
                     _neg_prompt_end,
                     _pos_prompt_end,
                     {},
                     start_seed,
                     end_seed,
                     start_denoising,
                     end_denoising,
                     guidance_scale,
                     guidance_scale,
                     num_inference_steps,
                     num_interpolation_steps,
                     mNumOutputSegments,
                     seed_image,
                     1.0f,
                     scheduler,
                     mycallback,
                     myIntcallback);
               }
               );

            mProgressFrac = 0.0;
            mProgMessage = "Running Music Generation";
            do {
               using namespace std::chrono_literals;
               status = generated_samples_future.wait_for(0.5s);
               {
                  std::lock_guard<std::mutex> guard(mProgMutex);
                  mProgress->SetMessage(TranslatableString{ wxString(mProgMessage), {} });

                  //this returns true if user clicks 'cancel'
                  if (TotalProgress(mProgressFrac))
                  {
                     mIsCancelled = true;
                  }
               }

            } while (status != std::future_status::ready);

            std::cout << "Running Riffusion Interpolation Pipeline complete" << std::endl;
            auto samples_pair = generated_samples_future.get();
            generated_samples_L = samples_pair.first;

            //if user cancelled, this condition will be true. 
            if (!generated_samples_L)
            {
               return false;
            }

            if (bStereo)
            {
               generated_samples_R = samples_pair.second;
            }


            if (generated_samples_L)
            {
               repair_audio_seams(generated_samples_L, mNumOutputSegments);
            }

            if (generated_samples_R)
            {
               repair_audio_seams(generated_samples_R, mNumOutputSegments);
            }
         }
         catch (const std::exception& error) {
            wxLogError("In Music Generation, exception: %s", error.what());
            EffectUIServices::DoMessageBox(*this,
               XO("Music Generation failed. See details in Help->Diagnostics->Show Log..."),
               wxICON_STOP,
               XO("Error"));
            return false;
         }

      }
      else
      {
         try
         {

            std::cout << "mNumInferenceSteps = " << mNumInferenceSteps << std::endl;
            std::cout << "mGuidanceScale = " << mGuidanceScale << std::endl;

            double gen_duration_in_secs = (225351.0 * 1) / 44100.0;
            settings.extra.SetDuration(gen_duration_in_secs);

            _pos_prompt_start = audacity::ToUTF8(mTextPrompt->GetLineText(0));
            std::cout << "prompt = " << _pos_prompt_start << std::endl;

            _neg_prompt_end = "";
            std::cout << "negative prompt = " << _neg_prompt_end << std::endl;

            std::string seed_str = audacity::ToUTF8(mSeed->GetLineText(0));
            _start_seed_str = seed_str;

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

            if (seed)
            {
               std::cout << "seed = " << *seed << std::endl;
            }

            std::string descriptor_str = "prompt: " + _pos_prompt_start;
            if (!_neg_prompt_end.empty() && _neg_prompt_end != "")
            {
               descriptor_str += ", negative prompt: " + _neg_prompt_end;
            }
            descriptor_str += ", seed: " + std::to_string(*seed);
            descriptor_str += ", scale: " + std::to_string(mGuidanceScale);
            descriptor_str += ", steps: " + std::to_string(mNumInferenceSteps);
            descriptor_str += ", scheduler: " + scheduler;

            added_trackName = wxString("Generated(Simple): " + descriptor_str);

            std::future< std::shared_ptr< cpp_stable_diffusion_ov::StableDiffusionPipeline > > sd_pipeline_creation_future = std::async(std::launch::async,
               [&riffusion_model_folder, &cache_path, &text_encoder_device, &unet_pos_device, &unet_neg_device,
               &vae_decoder_device, &vae_encoder_device] {
                  return std::make_shared< cpp_stable_diffusion_ov::StableDiffusionPipeline >(riffusion_model_folder,
                     cache_path,
                     text_encoder_device,
                     unet_pos_device,
                     unet_neg_device,
                     vae_decoder_device,
                     vae_encoder_device);
               }
            );

            float total_time = 0.f;
            std::future_status status;
            do {
               using namespace std::chrono_literals;
               status = sd_pipeline_creation_future.wait_for(0.5s);

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

               total_time += 0.5;
            } while (status != std::future_status::ready);

            std::shared_ptr< cpp_stable_diffusion_ov::StableDiffusionPipeline > sd_pipeline = sd_pipeline_creation_future.get();

            if (mIsCancelled)
            {
               return false;
            }

            std::pair< cpp_stable_diffusion_ov::CallbackFuncUnetIteration, void*> mycallback = { MyUnetCallback, this };
            std::future< std::shared_ptr<std::vector<uint8_t>> > generated_spec_future = std::async(std::launch::async,
               [this, &sd_pipeline, &scheduler, &seed, &mycallback] {
                  return (*sd_pipeline)(_pos_prompt_start,
                     _neg_prompt_end,
                     mNumInferenceSteps, //num 
                     scheduler,
                     seed, //seed
                     mGuidanceScale, //guidance scale
                     false, // don't give us BGR back
                     {},
                     mycallback);
               }
            );

            mProgressFrac = 0.0;
            mProgMessage = "Running Stable Diffusion Pipeline (Generating Spectrogram)";
            do {
               using namespace std::chrono_literals;
               status = generated_spec_future.wait_for(0.5s);
               {
                  std::lock_guard<std::mutex> guard(mProgMutex);
                  mProgress->SetMessage(TranslatableString{ wxString(mProgMessage), {} });
                  if (TotalProgress(mProgressFrac))
                  {
                     mIsCancelled = true;
                  }
               }

            } while (status != std::future_status::ready);

            auto spec = generated_spec_future.get();

            if (!spec)
            {
               return false;
            }

            std::cout << "Converting Spectrogram to Waveform" << std::endl;

            std::future< std::shared_ptr<std::vector<float>> > gen_sample_future_l = std::async(spec_to_wav,
               spec, 1);

            //generated_samples_L = spec_to_wav(spec, 1);

            std::future< std::shared_ptr<std::vector<float>> > gen_sample_future_r;
            if (bStereo)
            {
               gen_sample_future_r = std::async(spec_to_wav, spec, 2);
            }

            std::future_status l_status = std::future_status::timeout;
            std::future_status r_status = bStereo ? std::future_status::timeout : std::future_status::ready;
            do {
               mProgress->SetMessage(TranslatableString{ wxString("Converting Spectrogram to Waveform"), {} });

               using namespace std::chrono_literals;
               if (l_status != std::future_status::ready)
               {
                  l_status = gen_sample_future_l.wait_for(1s);
               }
               else if (r_status != std::future_status::ready)
               {
                  r_status = gen_sample_future_r.wait_for(1s);
               }
            } while ((l_status != std::future_status::ready) || (r_status != std::future_status::ready));

            generated_samples_L = gen_sample_future_l.get();
            if (!generated_samples_L)
               return false;
            if (bStereo)
            {
               generated_samples_R = gen_sample_future_r.get();
               if (!generated_samples_R)
                  return false;
            }
         }
         catch (const std::exception& error) {
            wxLogError("In Music Generation routine, exception: %s", error.what());
            EffectUIServices::DoMessageBox(*this,
               XO("Music Generation failed. See details in Help->Diagnostics->Show Log..."),
               wxICON_STOP,
               XO("Error"));
            return false;
         }
      }

      settings.extra.SetDuration(mDurationT->GetValue());
      const auto duration = settings.extra.GetDuration();

      //clip samples to max duration
      size_t max_output_samples = duration * 44100;
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

      int ntrack = 0;

      bool bNormalized = false;
      for (auto track : outputs.Get().Selected<WaveTrack>())
      {
            bool editClipCanMove = GetEditClipsCanMove();

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
                     tmpRight.Append((samplePtr)generated_samples_R->data(), floatSample, generated_samples_R->size());
                  }
               }

               //flush it
               auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();
               pTmpTrack->Flush();

               // The track we just populated with samples is 44100 Hz.
               // Within the ClearAndPaste() operation below, it will automatically
               // resample this to whatever the 'target' track sample rate is.
               pTmpTrack->SetRate(44100);

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

            ntrack++;
      }

      if (mIsCancelled)
      {
         return false;
      }

      if (bGoodResult ) {

         outputs.Commit();

         mT1 = mT0 + duration; // Update selection.
      }

      return bGoodResult;
   }
}

bool EffectOVMusicGeneration::UpdateProgress(double perc)
{
   if (!TotalProgress(perc / 100.0))
   {
      std::cout << "Total Progress returned false" << std::endl;
      return false;
   }

   return true;
}

std::unique_ptr<EffectEditor> EffectOVMusicGeneration::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess& access,
   const EffectOutputs*)
{
   DoPopulateOrExchange(S, access);
   return nullptr;
}

static int get_device_index(std::vector<EnumValueSymbol> enum_val_vector, std::string str)
{
   for (size_t i = 0; i < enum_val_vector.size(); i++)
   {
      auto vstr = audacity::ToUTF8(enum_val_vector[i].Msgid().MSGID().GET());
      if (vstr == str)
      {
         return (int)i;
      }
   }

   return 0;
}

void EffectOVMusicGeneration::DoPopulateOrExchange(
   ShuttleGui& S, EffectSettingsAccess& access)
{
   mUIParent = S.GetParent();

   //EnablePreview(false); //Port this
   
   S.StartVerticalLay(wxLEFT);
   {
      S.StartMultiColumn(3, wxLEFT);
      {
         mChoiceMode = S.Id(ID_Type_Mode)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_modeSelectionChoice)
            .AddChoice(XXO("Mode:"),
               Msgids(mGuiModeSelections.data(), mGuiModeSelections.size()));

         mUnloadModelsButton = S.Id(ID_Type_UnloadModelsButton).AddButton(XO("Unload Models"));
         if (!cpp_stable_diffusion_ov::ModelCollateralCache::instance()->CurrentUNetNegativeDevice())
         {
            mUnloadModelsButton->Enable(false);
         }
      }
      S.EndMultiColumn();

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

      S.StartMultiColumn(4, wxCENTER);
      {
            S.StartMultiColumn(2, wxCENTER);
            {
               mTextPromptStart = S.Id(ID_Type_StartPrompt)
                  .Style(wxTE_LEFT)
                  .AddTextBox(XXO("Start Prompt:"), wxString(_pos_prompt_start), 30);

               advancedSizer0 = mTextPromptStart->GetContainingSizer();

               mStartSeed = S.Id(ID_Type_StartSeed)
                  .Style(wxTE_LEFT)
                  .AddNumericTextBox(XXO("Seed:"), wxString(_start_seed_str), 10);

               auto t0 = S.Name(XO("Strength"))
                  .Validator<FloatingPointValidator<float>>(
                     6, &mStartDenoising,
                     NumValidatorStyle::NO_TRAILING_ZEROES,
                     0.0f,
                     1.0f)
                  .AddTextBox(XO("Strength"), L"", 12);

#if 0
               mNegativePromptAdvanced = S.Id(ID_Type_NegativePromptAdvanced)
                  .Style(wxTE_LEFT)
                  .AddTextBox(XXO("Negative Prompt:"), wxString(_neg_prompt_end), 30);
#endif

               mTypeChoiceSeedImage = S.Id(ID_Type_SeedImage)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_seedImageSelectionChoice)
                  .AddChoice(XXO("Seed Image:"),
                     Msgids(mGuiSeedImageSelections.data(), mGuiSeedImageSelections.size()));

               auto t1 = S.Name(XO("Guidance Scale"))
                  .Validator<FloatingPointValidator<float>>(
                     6, &mGuidanceScaleAdvanced,
                     NumValidatorStyle::NO_TRAILING_ZEROES,
                     0.0f,
                     10.0f)
                  .AddTextBox(XO("Guidance Scale"), L"", 12);

               mTextPromptNumInterpolationSteps = S.Id(ID_Type_NumInterpolationSteps)
                  .Validator<IntegerValidator<int>>(&mNumInterpolationSteps,
                     NumValidatorStyle::DEFAULT,
                     1,
                     100)
                  .AddTextBox(XO("Number of Interpolation Steps (5s per step)"), L"", 12);

               auto t4 = S.Name(XO("Num Inference Steps (For each interp. step)"))
                  .Validator<IntegerValidator<int>>(&mNumInferenceStepsAdvanced,
                     NumValidatorStyle::DEFAULT,
                     1,
                     100)
                  .AddTextBox(XO("Num Inference Steps (Per Interpolation Step)"), L"", 12);
            }
            S.EndMultiColumn();

           

            S.StartMultiColumn(2, wxCENTER);
            {
               mTextPromptEnd = S.Id(ID_Type_EndPrompt)
                  .Style(wxTE_LEFT)
                  .AddTextBox(XXO("End Prompt:"), wxString(_pos_prompt_end), 30);

               advancedSizer1 = mTextPromptEnd->GetContainingSizer();

               mEndSeed = S.Id(ID_Type_EndSeed)
                  .Style(wxTE_LEFT)
                  .AddNumericTextBox(XXO("Seed:"), wxString(_end_seed_str), 10);

               auto t = S.Name(XO("Strength"))
                  .Validator<FloatingPointValidator<float>>(
                     6, &mEndDenoising,
                     NumValidatorStyle::NO_TRAILING_ZEROES,
                     0.0f,
                     1.0f)
                  .AddTextBox(XO("Strength"), L"", 12);

            }
            S.EndMultiColumn();


            S.StartMultiColumn(2, wxCENTER);
            {
               mTextPrompt = S.Id(ID_Type_Prompt)
                  .Style(wxTE_LEFT)
                  .AddTextBox(XXO("What Kind of Music?"), wxString(_pos_prompt_start), 30);

               simpleSizer0 = mTextPrompt->GetContainingSizer();

#if 0
               mNegativePrompt = S.Id(ID_Type_NegativePrompt)
                  .Style(wxTE_LEFT)
                  .AddTextBox(XXO("Negative Prompt:"), wxString(_neg_prompt_end), 30);
#endif

               mTypeChoiceSeedImageSimple = S.Id(ID_Type_SeedImageSimple)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_seedImageSelectionChoiceSimple)
                  .AddChoice(XXO("Seed Image:"),
                     Msgids(mGuiSeedImageSelectionsSimple.data(), mGuiSeedImageSelectionsSimple.size()));

               mDenoisingSimpleCtl = S.Name(XO("Strength"))
                  .Validator<FloatingPointValidator<float>>(
                     6, &mDenoisingSimple,
                     NumValidatorStyle::NO_TRAILING_ZEROES,
                     0.0f,
                     1.0f)
                  .AddTextBox(XO("Strength"), L"", 12);

               mSeed = S.Id(ID_Type_Seed)
                  .Style(wxTE_LEFT)
                  .AddNumericTextBox(XXO("Seed:"), wxString(_start_seed_str), 10);

               auto t1 = S.Name(XO("Guidance Scale"))
                  .Validator<FloatingPointValidator<float>>(
                     6, &mGuidanceScale,
                     NumValidatorStyle::NO_TRAILING_ZEROES,
                     0.0f,
                     10.0f)
                  .AddTextBox(XO("Guidance Scale"), L"", 12);

               auto t3 = S.Name(XO("Num Inference Steps"))
                  .Validator<IntegerValidator<int>>(&mNumInferenceSteps,
                     NumValidatorStyle::DEFAULT,
                     1,
                     100)
                  .AddTextBox(XO("Num Inference Steps"), L"", 12);

               if (mTypeChoiceSeedImageSimple->GetSelection() == (mTypeChoiceSeedImageSimple->GetCount() - 1))
               {
                  mDenoisingSimpleCtl->Enable(false);
               }
               else
               {
                  mDenoisingSimpleCtl->Enable(true);
               }
            }
            S.EndMultiColumn();

            bool bAdvanced = (m_modeSelectionChoice == 1);
            advancedSizer0->ShowItems(bAdvanced);
            advancedSizer1->ShowItems(bAdvanced);
            simpleSizer0->ShowItems(!bAdvanced);

            S.StartMultiColumn(2, wxCENTER);
            {

               mTypeChoiceDeviceCtrl_TextEncoder = S.Id(ID_Type_TxtEncoder)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_deviceSelectionChoice_TextEncoder)
                  .AddChoice(XXO("Text Encoder Device:"),
                     Msgids(mGuiDeviceNonVPUSupportedSelections.data(), mGuiDeviceNonVPUSupportedSelections.size()));

               mTypeChoiceDeviceCtrl_UNetPositive = S.Id(ID_Type_UNetPositive)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_deviceSelectionChoice_UNetPositive)
                  .AddChoice(XXO("UNet + Device:"),
                     Msgids(mGuiDeviceVPUSupportedSelections.data(), mGuiDeviceVPUSupportedSelections.size()));

               mTypeChoiceDeviceCtrl_UNetNegative = S.Id(ID_Type_UNetNegative)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_deviceSelectionChoice_UNetNegative)
                  .AddChoice(XXO("UNet - Device:"),
                     Msgids(mGuiDeviceVPUSupportedSelections.data(), mGuiDeviceVPUSupportedSelections.size()));

               mTypeChoiceDeviceCtrl_VAEDecoder = S.Id(ID_Type_VAEDecoder)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_deviceSelectionChoice_VAEDecoder)
                  .AddChoice(XXO("VAE Decoder Device:"),
                     Msgids(mGuiDeviceNonVPUSupportedSelections.data(), mGuiDeviceNonVPUSupportedSelections.size()));

               mTypeChoiceDeviceCtrl_VAEEncoder = S.Id(ID_Type_VAEEncoder)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_deviceSelectionChoice_VAEEncoder)
                  .AddChoice(XXO("VAE Encoder Device:"),
                     Msgids(mGuiDeviceNonVPUSupportedSelections.data(), mGuiDeviceNonVPUSupportedSelections.size()));

               mTypeChoiceScheduler = S.Id(ID_Type_Scheduler)
                  .MinSize({ -1, -1 })
                  .Validator<wxGenericValidator>(&m_schedulerSelectionChoice)
                  .AddChoice(XXO("Scheduler:"),
                     Msgids(mGuiSchedulerSelections.data(), mGuiSchedulerSelections.size()));

               //mTypeChoiceScheduler->Hide();
            }
            S.EndMultiColumn();
      }
      S.EndMultiColumn();
   }

   

   S.EndVerticalLay();

}

void EffectOVMusicGeneration::OnChoice(wxCommandEvent& evt)
{
   if (mChoiceMode )
   {
      OnSimpleSeedImageChoice(evt);

      bool bAdvanced = (mChoiceMode->GetSelection() == 1);

      if (advancedSizer0)
      {
         advancedSizer0->ShowItems(bAdvanced);
         advancedSizer0->Layout();
      }

      if (advancedSizer1)
      {
         advancedSizer1->ShowItems(bAdvanced);
         advancedSizer1->Layout();
      }

      if (simpleSizer0)
      {
         simpleSizer0->ShowItems(!bAdvanced);
         simpleSizer0->Layout();
      }

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
   else
   {
      std::cout << "mChoiceMode does not exist!" << std::endl;
   }

}

void EffectOVMusicGeneration::OnUnloadModelsButtonClicked(wxCommandEvent& evt)
{
   cpp_stable_diffusion_ov::ModelCollateralCache::instance()->Reset();
   if (mUnloadModelsButton)
   {
      mUnloadModelsButton->Enable(false);
   }
}

void EffectOVMusicGeneration::OnSimpleSeedImageChoice(wxCommandEvent& evt)
{
   if (mChoiceMode)
   {
      bool bAdvanced = (mChoiceMode->GetSelection() == 1);

      double max_seconds = SECONDS_PER_GEN_SEGMENT;
      if (bAdvanced)
      {
         if (mTextPromptNumInterpolationSteps)
         {
            max_seconds = SECONDS_PER_GEN_SEGMENT;
            auto nsteps_str = audacity::ToUTF8(mTextPromptNumInterpolationSteps->GetLineText(0));
            if (!nsteps_str.empty() && nsteps_str != "")
            {
               int nsteps = std::stoi(nsteps_str);
               if (nsteps >= 1)
                  max_seconds = nsteps * SECONDS_PER_GEN_SEGMENT;
            }
            else
            {
               return;
            }
         }
         
      }
      else
      {
         max_seconds = DEFAULT_SIMPLE_NUM_INTERPOLATION_STEPS * SECONDS_PER_GEN_SEGMENT;
         auto seed_image = audacity::ToUTF8(mTypeChoiceSeedImageSimple->GetString(m_seedImageSelectionChoiceSimple));
         if (mTypeChoiceSeedImageSimple->GetSelection() == (mTypeChoiceSeedImageSimple->GetCount() - 1))
         {
            mDenoisingSimpleCtl->Enable(false);
            max_seconds = SECONDS_PER_GEN_SEGMENT;
         }
         else
         {
            mDenoisingSimpleCtl->Enable(true);
         }
      }

      if (mDurationT->GetValue() > max_seconds)
      {
         mDurationT->SetValue(max_seconds);
      }
   }
}

bool EffectOVMusicGeneration::TransferDataToWindow(const EffectSettings &settings)
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

bool EffectOVMusicGeneration::TransferDataFromWindow(EffectSettings & settings)
{

   if (!mUIParent->Validate() || !mUIParent->TransferDataFromWindow())
   {
      return false;
   }


   settings.extra.SetDuration(mDurationT->GetValue());
   //EnablePreview(false);  //Port this
   return true;
}
