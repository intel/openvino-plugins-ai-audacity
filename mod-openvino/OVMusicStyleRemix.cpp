// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include <openvino/openvino.hpp>
#include "OVMusicStyleRemix.h"
#include "WaveTrack.h"
#include "EffectOutputTracks.h"
#include "effects/EffectEditor.h"

#include <math.h>
#include <iostream>
#include <time.h>
#include <future>


#include "BasicUI.h"
#include "ViewInfo.h"
#include "TimeWarper.h"
#include "LoadEffects.h"
#include "htdemucs.h"
#include "widgets/NumericTextCtrl.h"

#include <wx/intl.h>
#include <wx/valgen.h>
#include <wx/textctrl.h>
#include <wx/wrapsizer.h>
#include <wx/button.h>

#include "ShuttleGui.h"

#include "widgets/valnum.h"
#include <wx/choice.h>
#include "FileNames.h"
#include "CodeConversions.h"
#include "SyncLock.h"

#include "InterpolateAudio.h"

#include "Mix.h"
#include "MixAndRender.h"

#include "cpp_stable_diffusion_audio_ov/riffusion_audio_to_audio_pipeline.h"
#include "cpp_stable_diffusion_ov/model_collateral_cache.h"

#include "OVStringUtils.h"

const ComponentInterfaceSymbol EffectOVMusicStyleRemix::Symbol{ XO("OpenVINO Music Style Remix") };

namespace { BuiltinEffectsModule::Registration< EffectOVMusicStyleRemix > reg; }

BEGIN_EVENT_TABLE(EffectOVMusicStyleRemix, wxEvtHandler)
    EVT_BUTTON(ID_Type_UnloadModelsButton, EffectOVMusicStyleRemix::OnUnloadModelsButtonClicked)
END_EVENT_TABLE()

EffectOVMusicStyleRemix::EffectOVMusicStyleRemix()
{
   Parameters().Reset(*this);

   ov::Core core;
   auto devices = core.get_available_devices();

   for (auto d : devices)
   {
      //GNA devices are not supported
      if (d.find("GNA") != std::string::npos) continue;

      //mTypeChoiceDeviceCtrl_UNetPositive.push_back()
      mGuiDeviceVPUSupportedSelections.push_back(wxString(d));

      if (d == "NPU")
      {
         m_deviceSelectionChoice_UNetNegative = mGuiDeviceVPUSupportedSelections.size() - 1;
         continue;
      }

      mGuiDeviceNonVPUSupportedSelections.push_back(wxString(d));

      if (d == "GPU")
      {
         m_deviceSelectionChoice_VAEEncoder = mGuiDeviceNonVPUSupportedSelections.size() - 1;
         m_deviceSelectionChoice_VAEDecoder = mGuiDeviceNonVPUSupportedSelections.size() - 1;
         m_deviceSelectionChoice_UNetPositive = mGuiDeviceNonVPUSupportedSelections.size() - 1;
      }
   }

   //audacity::ToUTF8(mTypeChoiceDeviceCtrl_TextEncoder->GetString(m_deviceSelectionChoice_TextEncoder));


   std::vector<std::string> schedulerChoices = { "EulerDiscreteScheduler", "USTMScheduler", "PNDMScheduler" };
   for (auto d : schedulerChoices)
   {
      mGuiSchedulerSelections.push_back({ TranslatableString{ wxString(d), {}} });
   }
}

EffectOVMusicStyleRemix::~EffectOVMusicStyleRemix()
{
   cpp_stable_diffusion_ov::ModelCollateralCache::instance()->Reset();
}


// ComponentInterface implementation

ComponentInterfaceSymbol EffectOVMusicStyleRemix::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVMusicStyleRemix::GetDescription() const
{
   return XO("Remixes an audio track, given a text prompt");
}

VendorSymbol EffectOVMusicStyleRemix::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

// EffectDefinitionInterface implementation

EffectType EffectOVMusicStyleRemix::GetType() const
{
   return EffectTypeProcess;
}

bool EffectOVMusicStyleRemix::segment_callback(size_t num_segments_complete, size_t total_segments)
{
   //mInterp_steps_complete = interp_step_i_complete + 1;
   if (num_segments_complete == 0)
   {
      mNumSegments = total_segments;
   }

   mSegments_complete = num_segments_complete;

   return true;
}

bool EffectOVMusicStyleRemix::unet_callback(size_t unet_step_i_complete, size_t unet_total_iterations)
{
   std::lock_guard<std::mutex> guard(mProgMutex);

   mProgMessage = "Audio Segment " + std::to_string(mSegments_complete + 1) + " / " + std::to_string(mNumSegments) +
      ": Running UNet iteration " + std::to_string(unet_step_i_complete + 1) + " / " + std::to_string(unet_total_iterations);

   size_t total_iterations = unet_total_iterations * mNumSegments;
   size_t iterations_complete = (mSegments_complete * unet_total_iterations) + (unet_step_i_complete + 1);

   if (iterations_complete == total_iterations)
   {
      mProgMessage = " Finalizing Waveform...";
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

static bool SegmentCompleteCallback(size_t num_segments_complete,
   size_t num_total_segments,
   std::shared_ptr<std::vector<float>> wav,
   std::shared_ptr<std::vector<uint8_t>> img_rgb,
   size_t img_width,
   size_t img_height,
   void* user)
{
   //std::cout << "SegmentCompleteCallback: " <<
   //   ((double)num_segments_complete / (double)num_total_segments) * 100 << "% complete." << std::endl;
   EffectOVMusicStyleRemix* r = (EffectOVMusicStyleRemix*)user;
   return r->segment_callback(num_segments_complete, num_total_segments);
}

static bool MyUnetCallback(size_t unet_step_i_complete,
   size_t num_unet_steps,
   void* user)
{
   //std::cout << "unet iteration " << unet_step_i_complete << " / " << num_unet_steps << " complete." << std::endl;
   EffectOVMusicStyleRemix* r = (EffectOVMusicStyleRemix*)user;
   return r->unet_callback(unet_step_i_complete, num_unet_steps);
}

static std::shared_ptr< cpp_stable_diffusion_ov::RiffusionAudioToAudioPipeline > create_a2a_pipeline(std::string riffusion_model_folder,
   std::string cache_path, std::string text_encoder_device, std::string unet_pos_device, std::string unet_neg_device,
   std::string vae_decoder_device, std::string vae_encoder_device)
{
   return std::make_shared< cpp_stable_diffusion_ov::RiffusionAudioToAudioPipeline >(riffusion_model_folder, cache_path,
      text_encoder_device, unet_pos_device, unet_neg_device, vae_decoder_device, vae_encoder_device);
}


static std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > run_a2a_pipeline(std::shared_ptr< cpp_stable_diffusion_ov::RiffusionAudioToAudioPipeline > pipeline,
   float* pInput_44100_wav,
   float* pInput_44100_wav_R,
   size_t nsamples_to_riffuse,
   size_t ntotal_samples_allowed_to_read,
   const std::string prompt,
   std::optional< std::string > negative_prompt,
   int num_inference_steps,
   const std::string& scheduler_str,
   std::optional< unsigned int > seed,
   float guidance_scale,
   float denoising_strength,
   float crossfade_overlap_seconds,
   std::optional<std::pair< cpp_stable_diffusion_ov::CallbackFuncUnetIteration, void*>> unet_iteration_callback,
   std::optional<std::pair<cpp_stable_diffusion_ov:: RiffusionAudioToAudioPipeline::CallbackFuncAudioSegmentComplete, void*>> segment_callback)
{
   return (*pipeline)(pInput_44100_wav, pInput_44100_wav_R, nsamples_to_riffuse, ntotal_samples_allowed_to_read,
      prompt, negative_prompt, num_inference_steps, scheduler_str, seed, guidance_scale, denoising_strength, crossfade_overlap_seconds,
      unet_iteration_callback, segment_callback);
}

bool EffectOVMusicStyleRemix::Process(EffectInstance&, EffectSettings& settings)
{
   bool bGoodResult = true;

   mIsCancelled = false;

   try
   {
      EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };

      // Determine the total time (in samples) used by all of the target tracks
      sampleCount totalTime = 0;

      FilePath model_folder = FileNames::MkDir(wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath());
      std::string riffusion_model_folder = audacity::ToUTF8(wxFileName(model_folder, wxString("riffusion-unet-quantized-int8"))
         .GetFullPath());

      std::cout << "riffusion_model_folder = " << riffusion_model_folder << std::endl;

      FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());

      //Note: Using a variant of wstring conversion that seems to work more reliably when there are special characters present in the path.
      std::string cache_path = wstring_to_string(wxFileName(cache_folder).GetFullPath().ToStdWstring());

      std::cout << "cache path = " << cache_path << std::endl;

      std::cout << "Creating pipeline object with following devices" << std::endl;
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

      std::pair< cpp_stable_diffusion_ov::CallbackFuncUnetIteration, void*> unet_callback = { MyUnetCallback, this };
      std::pair< cpp_stable_diffusion_ov::RiffusionAudioToAudioPipeline::CallbackFuncAudioSegmentComplete, void*> seg_callback = { SegmentCompleteCallback, this };

      _pos_prompt = audacity::ToUTF8(mTextPrompt->GetLineText(0));

      std::cout << "start prompt = " << _pos_prompt << std::endl;
      std::cout << "mDenoisingStrength = " << mDenoisingStrength << std::endl;

      std::cout << "mNumInferenceSteps = " << mNumInferenceSteps << std::endl;

      //_neg_prompt = audacity::ToUTF8(mNegativePrompt->GetLineText(0));
      _neg_prompt = "";

      std::cout << "negative prompt = " << _neg_prompt << std::endl;

      auto scheduler = audacity::ToUTF8(mTypeChoiceScheduler->GetString(m_schedulerSelectionChoice));
      std::cout << "scheduler = " << scheduler << std::endl;

      std::cout << "mGuidanceScale = " << mGuidanceScale << std::endl;


      unsigned int seed;

      _seed_str = audacity::ToUTF8(mSeed->GetLineText(0));
      if (!_seed_str.empty() && _seed_str != "")
      {
         seed = std::stoul(_seed_str);
      }
      else
      {
         //seed is not set.. set it to time.
         time_t t;
         seed = (unsigned)time(&t);
      }

      if (seed)
      {
         std::cout << "seed = " << seed << std::endl;
      }

      std::string descriptor_str = "prompt: " + _pos_prompt;
      if (!_neg_prompt.empty() && _neg_prompt != "")
      {
         descriptor_str += ", negative prompt: " + _neg_prompt;
      }
      descriptor_str += ", seed: " + std::to_string(seed);
      descriptor_str += ", strength: " + std::to_string(mDenoisingStrength);
      descriptor_str += ", scale: " + std::to_string(mGuidanceScale);
      descriptor_str += ", steps: " + std::to_string(mNumInferenceSteps);
      descriptor_str += ", scheduler: " + scheduler;

      std::shared_ptr< cpp_stable_diffusion_ov::RiffusionAudioToAudioPipeline > pipeline;

      {
         std::future<std::shared_ptr< cpp_stable_diffusion_ov::RiffusionAudioToAudioPipeline >> riff_pipeline_creation_future;

         riff_pipeline_creation_future = std::async(create_a2a_pipeline, riffusion_model_folder, cache_path,
            text_encoder_device, unet_pos_device, unet_neg_device, vae_decoder_device, vae_encoder_device);

         std::future_status status;

         float total_time = 0.f;
         do {
            using namespace std::chrono_literals;
            status = riff_pipeline_creation_future.wait_for(0.5s);

            {
               std::string message = "Loading Music Style Remix AI Models... ";
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

         pipeline = riff_pipeline_creation_future.get();
      }

      if (mIsCancelled)
      {
         return false;
      }

      std::vector< TrackListHolder > tracks_to_process;
      std::vector< double > selection_lengths;
      std::vector< std::pair<float, float> > orig_rms;

      for (auto track : outputs.Get().Selected<WaveTrack>())
      {
         auto left = track->GetChannel(0);
         if (track->Channels().size() > 1)
         {
            auto right = track->GetChannel(1);

            double selection_length;
            auto start = wxMin(left->GetStartTime(), right->GetStartTime());
            auto end = wxMax(left->GetEndTime(), right->GetEndTime());
            double mCurT0 = mT0 < start ? start : mT0;

            double modified_t1 = mT1 + (225351.0 / 44100.0);
            double mCurT1 = modified_t1 > end ? end : modified_t1;

            {
               double selectionT1 = mT1 > end ? end : mT1;
               selection_length = selectionT1 - mCurT0;
            }

            float left_rms = left->GetRMS(mCurT0, mCurT1);
            float right_rms = right->GetRMS(mCurT0, mCurT1);

            orig_rms.push_back({ left_rms, right_rms });

            selection_lengths.push_back(selection_length);

            auto start_s = left->TimeToLongSamples(mCurT0);
            auto end_s = left->TimeToLongSamples(mCurT1);
            size_t total_samples = (end_s - start_s).as_size_t();
            Floats entire_input{ total_samples };

            {
               // create a temporary track list to append samples to
               auto tmp_tracklist = track->WideEmptyCopy();

               auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();

               auto iter =
                  pTmpTrack->Channels().begin();

               bool bOkay = left->GetFloats(entire_input.get(), start_s, total_samples);
               if (!bOkay)
               {
                  throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
                     std::to_string(total_samples) + "samples");
               }

               //append output samples to L & R channels.
               auto& tmpLeft = **iter++;
               tmpLeft.Append((samplePtr)entire_input.get(), floatSample, total_samples);

               bOkay = right->GetFloats(entire_input.get(), start_s, total_samples);
               if (!bOkay)
               {
                  throw std::runtime_error("unable to get all right samples. GetFloats() failed for " +
                     std::to_string(total_samples) + " samples");
               }

               auto& tmpRight = **iter;
               tmpRight.Append((samplePtr)entire_input.get(), floatSample, total_samples);

               //flush it
               pTmpTrack->Flush();

               if (pTmpTrack->GetRate() != 44100)
               {
                  pTmpTrack->Resample(44100, mProgress);
               }

               tracks_to_process.push_back(tmp_tracklist);
            }
         }
         else
         {
            //mono
            auto start = left->GetStartTime();
            auto end = left->GetEndTime();

            double mCurT0 = mT0 < start ? start : mT0;

            double modified_t1 = mT1 + (225351.0 / 44100.0);
            double mCurT1 = modified_t1 > end ? end : modified_t1;

            {
               double selectionT1 = mT1 > end ? end : mT1;

               double selection_length = selectionT1 - mCurT0;

               selection_lengths.push_back(selection_length);
            }

            float left_rms = left->GetRMS(mCurT0, mCurT1);

            orig_rms.push_back({ left_rms, left_rms });

            auto start_s = left->TimeToLongSamples(mCurT0);
            auto end_s = left->TimeToLongSamples(mCurT1);
            size_t total_samples = (end_s - start_s).as_size_t();

            // create a temporary track list to append samples to
            auto tmp_tracklist = track->WideEmptyCopy();

            auto iter =
               (*tmp_tracklist->Any<WaveTrack>().begin())->Channels().begin();

            Floats entire_input{ total_samples };
            bool bOkay = left->GetFloats(entire_input.get(), start_s, total_samples);
            if (!bOkay)
            {
               throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
                  std::to_string(total_samples) + " samples");
            }

            auto& tmpLeft = **iter++;
            tmpLeft.Append((samplePtr)entire_input.get(), floatSample, total_samples);

            //flush it
            auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();
            pTmpTrack->Flush();

            if (pTmpTrack->GetRate() != 44100)
            {
               pTmpTrack->Resample(44100, mProgress);
            }

            tracks_to_process.push_back(tmp_tracklist);
         }
      }

      if (!bGoodResult)
         return bGoodResult;

      mProgress->SetMessage(XO("Running Music Style Remix"));

      for (size_t ti = 0; ti < tracks_to_process.size(); ti++)
      {
         auto pTrack = *(tracks_to_process[ti])->Any<WaveTrack>().begin();

         auto left_track = pTrack->GetChannel(0);
         std::shared_ptr<WaveChannel> right_track = {};
         if (pTrack->Channels().size() > 1)
         {
            right_track = pTrack->GetChannel(1);
         }

         double selection_length = selection_lengths[ti];
         sampleFormat origFmt = pTrack->GetSampleFormat();

         double trackStart = pTrack->GetStartTime();
         double trackEnd = pTrack->GetEndTime();

         auto start = left_track->TimeToLongSamples(trackStart);
         auto end = left_track->TimeToLongSamples(trackEnd);
         auto selection_end = left_track->TimeToLongSamples(trackStart + selection_length);

         size_t selection_samples = (selection_end - start).as_size_t();
         size_t total_samples = (end - start).as_size_t();

         Floats entire_input_left{ total_samples };
         bGoodResult = left_track->GetFloats(entire_input_left.get(), start, total_samples);
         if (!bGoodResult)
         {
            throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
               std::to_string(total_samples) + " samples");
         }

         Floats entire_input_right;
         if (right_track)
         {
            entire_input_right = Floats{ total_samples };
            bGoodResult = right_track->GetFloats(entire_input_right.get(), start, total_samples);
            if (!bGoodResult)
            {
               throw std::runtime_error("unable to get all right samples. GetFloats() failed for " +
                  std::to_string(total_samples) + " samples");
            }
         }

         std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > out_samples;
         {

            std::future<std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> >> riff_pipeline_run_future;

            std::optional< std::string > neg_prompt;
            riff_pipeline_run_future = std::async(run_a2a_pipeline, pipeline, entire_input_left.get(),
               right_track ? entire_input_right.get() : nullptr,
               selection_samples,
               total_samples,
               _pos_prompt,
               neg_prompt,
               mNumInferenceSteps,
               scheduler,
               seed,
               mGuidanceScale,
               mDenoisingStrength,
               0.2f,
               unet_callback,
               seg_callback);

            std::future_status status;

            mProgressFrac = 0.0;
            mProgMessage = "Running Music Style Remix";
            do {
               using namespace std::chrono_literals;
               status = riff_pipeline_run_future.wait_for(0.5s);
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

            out_samples = riff_pipeline_run_future.get();
         }

         //this condition will be true upon cancellation
         if (!out_samples.first)
         {
            return false;
         }

         //super inefficient, but we populate the track to measure RMS, perform normalization, and then
         // re-populate it.
         {
            //super inefficient, but we populate the track to measure RMS, perform normalization, and then
            // re-populate it.
            {
               // create a temporary track list to append samples to
               auto tmp_tracklist = pTrack->WideEmptyCopy();
               auto iter =
                  (*tmp_tracklist->Any<WaveTrack>().begin())->Channels().begin();

               //append output samples to L & R channels.
               auto& tmpLeft = **iter++;
               tmpLeft.Append((samplePtr)out_samples.first->data(), floatSample, out_samples.first->size());

               auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();
               pTmpTrack->Flush();

               float new_left_rms = tmpLeft.GetRMS(tmpLeft.GetStartTime(), tmpLeft.GetEndTime());
               std::cout << "new_left_rms = " << new_left_rms << std::endl;

               float rms_ratio = orig_rms[ti].first / new_left_rms;
               std::cout << "rms_ratio (left) = " << rms_ratio << std::endl;

               {
                  float* pLeft = out_samples.first->data();
                  for (size_t si = 0; si < out_samples.first->size(); si++)
                  {
                     pLeft[si] *= rms_ratio;
                  }
               }

               if (right_track)
               {
                  auto& tmpRight = **iter++;
                  tmpRight.Append((samplePtr)out_samples.second->data(), floatSample, out_samples.second->size());

                  pTmpTrack->Flush();

                  float new_right_rms = tmpRight.GetRMS(tmpRight.GetStartTime(), tmpRight.GetEndTime());
                  rms_ratio = orig_rms[ti].second / new_right_rms;

                  {
                     float* pRight = out_samples.second->data();
                     for (size_t si = 0; si < out_samples.second->size(); si++)
                     {
                        pRight[si] *= rms_ratio;
                     }
                  }
               }
            }
         }

         //okay, now actually populate & push new output track
         {
            wxString added_trackName = wxString("-Remixed: ") + wxString(descriptor_str);

            auto orig_track_name = pTrack->GetName();

            // Workaround for 3.4.X issue where setting name of a new output track
            // retains the label of the track that it was copied from. So, we'll
            // change the name of the input track here, copy it, and then change it
            // back later.
            pTrack->SetName(orig_track_name + added_trackName);

            //Create new output track from input track.
            //WaveTrack::Holder newOutputTrack = inputTrack->EmptyCopy();
            auto newOutputTrackList = pTrack->WideEmptyCopy();

            auto newOutputTrack = *newOutputTrackList->Any<WaveTrack>().begin();

            // append the remix info to the track name
            // TODO: For Audacity 3.4, this doesn't seem to work as expected.
            // The generated track will not have this name.
            //newOutputTrack->SetName(orig_track_name + added_trackName);

            // create a temporary track list to append samples to
            auto tmp_tracklist = pTrack->WideEmptyCopy();
            auto iter =
               (*tmp_tracklist->Any<WaveTrack>().begin())->Channels().begin();

            //append output samples to L & R channels.
            auto& tmpLeft = **iter++;
            tmpLeft.Append((samplePtr)out_samples.first->data(), floatSample, out_samples.first->size());

            if (right_track)
            {
               auto& tmpRight = **iter;
               tmpRight.Append((samplePtr)out_samples.second->data(), floatSample, out_samples.second->size());
            }

            //flush it
            auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();
            pTmpTrack->Flush();

            auto pProject = FindProject();
            const auto& selectedRegion =
               ViewInfo::Get(*pProject).selectedRegion;

            // Clear & Paste into new output track
            newOutputTrack->ClearAndPaste(selectedRegion.t0() - pTmpTrack->GetStartTime(),
               selectedRegion.t1() - pTmpTrack->GetStartTime(), *tmp_tracklist);

            if (TracksBehaviorsSolo.ReadEnum() == SoloBehaviorSimple)
            {
               //If in 'simple' mode, if original track is solo,
               // mute the new track and set it to *not* be solo.
               if (newOutputTrack->GetSolo())
               {
                  newOutputTrack->SetMute(true);
                  newOutputTrack->SetSolo(false);
               }
            }

            //auto v = *newOutputTrackList;
            // Add the new track to the output.
            outputs.AddToOutputTracks(std::move(*newOutputTrackList));

            // Change name back to original. Part of workaround described above.
            pTrack->SetName(orig_track_name);
         }
      }

      if (mIsCancelled)
      {
         return false;
      }

      if (bGoodResult)
         outputs.Commit();
   }
   catch (const std::exception& error) {
      std::cout << "Music Style Remix routine failed: " << error.what() << std::endl;
      bGoodResult = false;
   }

   
   return bGoodResult;
}

bool EffectOVMusicStyleRemix::UpdateProgress(double perc)
{
   if (!TotalProgress(perc / 100.0))
   {
      std::cout << "Total Progress returned false" << std::endl;
      return false;
   }

   return true;
}

std::unique_ptr<EffectEditor> EffectOVMusicStyleRemix::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess& access,
   const EffectOutputs*)
{
   DoPopulateOrExchange(S, access);
   return nullptr;
}

void EffectOVMusicStyleRemix::DoPopulateOrExchange(
   ShuttleGui& S, EffectSettingsAccess& access)
{
   mUIParent = S.GetParent();

   S.StartMultiColumn(1, wxLEFT);
   {
      mUnloadModelsButton = S.Id(ID_Type_UnloadModelsButton).AddButton(XO("Unload Models"));
      if (!cpp_stable_diffusion_ov::ModelCollateralCache::instance()->CurrentUNetNegativeDevice())
      {
         mUnloadModelsButton->Enable(false);
      }
   }
   S.EndMultiColumn();

   S.StartMultiColumn(3, wxCENTER);
   {
      S.StartMultiColumn(2, wxCENTER);
      {
         mTextPrompt = S.Id(ID_Type_StartPrompt)
            .Style(wxTE_LEFT)
            .AddTextBox(XXO("What to remix to?"), wxString(_pos_prompt), 30);

#if 0
         mNegativePrompt = S.Id(ID_Type_NegativePrompt)
            .Style(wxTE_LEFT)
            .AddTextBox(XXO("Negative Prompt:"), wxString(_neg_prompt), 30);
#endif

         auto t0 = S.Name(XO("Num Inference Steps (Per ~5 second clip)"))
            .Validator<IntegerValidator<int>>(&mNumInferenceSteps,
               NumValidatorStyle::DEFAULT,
               1,
               100)
            .AddTextBox(XO("Num Inference Steps (Per ~5 second clip)"), L"", 12);

         auto t1 = S.Name(XO("Guidance Scale"))
            .Validator<FloatingPointValidator<float>>(
               6, &mGuidanceScale,
               NumValidatorStyle::NO_TRAILING_ZEROES,
               0.0f,
               10.0f)
            .AddTextBox(XO("Guidance Scale"), L"", 12);

         mSeed = S.Id(ID_Type_Seed)
            .Style(wxTE_LEFT)
            .AddNumericTextBox(XXO("Seed:"), wxString(_seed_str), 10);

         auto t2 = S.Name(XO("Strength"))
            .Validator<FloatingPointValidator<float>>(
               6, &mDenoisingStrength,
               NumValidatorStyle::NO_TRAILING_ZEROES,
               0.0f,
               1.0f)
            .AddTextBox(XO("Strength"), L"", 12);
      }
      S.EndMultiColumn();

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

      }
      S.EndMultiColumn();
   }
   //S.EndHorizontalLay();
   S.EndMultiColumn();
}

bool EffectOVMusicStyleRemix::TransferDataToWindow(const EffectSettings&)
{
   if (!mUIParent || !mUIParent->TransferDataToWindow())
   {
      return false;
   }

   EffectEditor::EnablePreview(mUIParent, false);

   return true;
}

void EffectOVMusicStyleRemix::OnUnloadModelsButtonClicked(wxCommandEvent& evt)
{
   cpp_stable_diffusion_ov::ModelCollateralCache::instance()->Reset();
   if (mUnloadModelsButton)
   {
      mUnloadModelsButton->Enable(false);
   }
}


bool EffectOVMusicStyleRemix::TransferDataFromWindow(EffectSettings& settings)
{
   if (!mUIParent || !mUIParent->Validate() || !mUIParent->TransferDataFromWindow())
   {
      return false;
   }

   return true;
}
