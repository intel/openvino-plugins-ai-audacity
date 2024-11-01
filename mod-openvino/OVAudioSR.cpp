// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "OVAudioSR.h"
#include "WaveTrack.h"
#include "WaveChannelUtilities.h"
#include "EffectOutputTracks.h"
#include "effects/EffectEditor.h"
#include <math.h>
#include <iostream>
#include <wx/log.h>

#include "BasicUI.h"
#include "ViewInfo.h"
#include "TimeWarper.h"
#include "LoadEffects.h"
#include "audio_sr.h"

#include "biquad.h"

#include <wx/intl.h>
#include <wx/valgen.h>
#include <wx/checkbox.h>
#include <wx/sizer.h>

#include "ShuttleGui.h"
#include <wx/choice.h>
#include "FileNames.h"
#include "CodeConversions.h"
#include <future>

#include "widgets/valnum.h"

#include "OVStringUtils.h"
#include <openvino/openvino.hpp>

const ComponentInterfaceSymbol EffectOVAudioSR::Symbol{ XO("OpenVINO Super Resolution") };

namespace { BuiltinEffectsModule::Registration< EffectOVAudioSR > reg; }

BEGIN_EVENT_TABLE(EffectOVAudioSR, wxEvtHandler)
EVT_CHECKBOX(ID_Type_AdvancedCheckbox, EffectOVAudioSR::OnAdvancedCheckboxChanged)
EVT_BUTTON(ID_Type_DeviceInfoButton, EffectOVAudioSR::OnDeviceInfoButtonClicked)
END_EVENT_TABLE()

EffectOVAudioSR::EffectOVAudioSR()
{
   ov::Core core;

   mSupportedDevices = core.get_available_devices();

   for (auto d : mSupportedDevices)
   {
      mGuiDeviceSelections.push_back({ TranslatableString{ wxString(d), {}} });

      m_simple_to_full_device_map.push_back({ d, core.get_property(d, "FULL_DEVICE_NAME").as<std::string>() });
   }

   mGuiModelSelections.push_back({ TranslatableString{ wxString("Basic (General)"), {}} });
   mGuiModelSelections.push_back({ TranslatableString{ wxString("Speech"), {}} });
}

EffectOVAudioSR::~EffectOVAudioSR()
{

}

// ComponentInterface implementation
ComponentInterfaceSymbol EffectOVAudioSR::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVAudioSR::GetDescription() const
{
   return XO("Performs Super Resolution upscaling to 48 khz");
}

VendorSymbol EffectOVAudioSR::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

unsigned EffectOVAudioSR::GetAudioInCount() const
{
   return 2;
}

// EffectDefinitionInterface implementation

EffectType EffectOVAudioSR::GetType() const
{
   return EffectTypeProcess;
}

bool EffectOVAudioSR::IsInteractive() const
{
   return true;
}

std::unique_ptr<EffectEditor> EffectOVAudioSR::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess&,
   const EffectOutputs*)
{
   mUIParent = S.GetParent();

   S.AddSpace(0, 5);
   S.StartVerticalLay();
   {
      S.StartMultiColumn(2, wxLEFT);
      {
         //m_deviceSelectionChoice
         mTypeChoiceModelCtrl = S.Id(ID_Type)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_modelSelectionChoice)
            .AddChoice(XXO("Model:"),
               Msgids(mGuiModelSelections.data(), mGuiModelSelections.size()));
      }
      S.EndMultiColumn();

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

      //advanced options
      S.StartMultiColumn(2, wxLEFT);
      {
         mShowAdvancedOptionsCheckbox = S.Id(ID_Type_AdvancedCheckbox).AddCheckBox(XXO("&Advanced Options"), false);
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         mNumberOfShiftsCtrl = S.Name(XO("Shifts"))
            .Validator<IntegerValidator<int>>(&mNumberOfShifts,
               NumValidatorStyle::DEFAULT,
               1,
               8)
            .AddTextBox(XO("Shifts"), L"", 12);

         advancedSizer = mNumberOfShiftsCtrl->GetContainingSizer();
      }
      S.EndMultiColumn();

   }
   S.EndVerticalLay();

   show_or_hide_advanced_options();

   return nullptr;
}

void EffectOVAudioSR::show_or_hide_advanced_options()
{
   if (advancedSizer)
   {
      advancedSizer->ShowItems(mbAdvanced);
      advancedSizer->Layout();
   }
}

void EffectOVAudioSR::OnAdvancedCheckboxChanged(wxCommandEvent& evt)
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

void EffectOVAudioSR::OnDeviceInfoButtonClicked(wxCommandEvent& evt)
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

static std::vector<WaveTrack::Holder> CreateSourceTracks
(WaveTrack* leader, std::vector<std::string>& labels)
{
   std::vector<WaveTrack::Holder> sources;
   for (auto& label : labels)
   {
      WaveTrack::Holder srcTrack = leader->EmptyCopy();

      srcTrack->SetSelected(false);
      srcTrack->SetSolo(false);

      // append the source name to the track's name
      srcTrack->SetName(srcTrack->GetName() + wxString("-" + label));
      sources.emplace_back(srcTrack);
   }
   return sources;
}

static bool AudioSRCallback(int ith_step_complete, void* user)
{
   EffectOVAudioSR* audio_sr = (EffectOVAudioSR*)user;
   return audio_sr->StepComplete(ith_step_complete);
}

bool EffectOVAudioSR::StepComplete(int ith_step)
{
   std::lock_guard<std::mutex> guard(mProgMutex);

   _ddim_steps_complete++;
   if (_total_ddim_steps > 0)
   {
      float perc = (float)_ddim_steps_complete / (float)_total_ddim_steps;
      mProgressFrac = perc;

      if (mIsCancelled)
      {
         return false;
      }
   }

   return true;
}

static inline std::shared_ptr<std::vector<float>> sos_lowpass_filter(float *pIn, size_t nsamples, double cutoff, double fs)
{
   std::cout << "sos_lowpass_filter called with cutoff=" << cutoff << std::endl;

   int order = 8;

   double nyquist = fs / 2.0;

   auto biquad = ovaudiosr::Biquad::CalcButterworthFilter(order, nyquist, cutoff, ovaudiosr::Biquad::kLowPass);

   for (int iPair = 0; iPair < ((order + 1) / 2); iPair++)
   {
      biquad[iPair].Reset();
   }

   auto output = std::make_shared< std::vector< float > >(nsamples);

   //forward pass.
   {
      float* ibuf = pIn;
      float* obuf = output->data();
      for (int iPair = 0; iPair < ((order + 1) / 2); iPair++)
      {
         biquad[iPair].Process(ibuf, obuf, (int)nsamples);
         ibuf = obuf;
      }
   }

   //reverse pass
   std::reverse(output->begin(), output->end());

   for (int iPair = 0; iPair < ((order + 1) / 2); iPair++)
   {
      biquad[iPair].Reset();
   }

   {
      float* obuf = output->data();
      float* ibuf = obuf;
      for (int iPair = 0; iPair < ((order + 1) / 2); iPair++)
      {
         biquad[iPair].Process(ibuf, obuf, nsamples);
         ibuf = obuf;
      }
   }

   //reverse it back
   std::reverse(output->begin(), output->end());

   return output;
}


static inline double calc_rms( float *pSamples, size_t nsamples )
{
   double sumOfSquares = 0.0;
   for (int i = 0; i < nsamples; i++)
   {
      float sample = pSamples[i];
      sumOfSquares += sample * sample; // Square each sample
   }

   double meanOfSquares = sumOfSquares / nsamples;
   double rms = std::sqrt(meanOfSquares);
   return rms;
}

static inline std::shared_ptr< std::vector<float> > normalize_pad_lowpass(float* pInput, size_t nsamples, WaveTrack::Holder pTrack, std::shared_ptr<AudioSR> audioSR, Batch& batch)
{
   audioSR->normalize_and_pad(pInput, nsamples, batch);

   // Take the normalized waveform, and apply a lowpass filter to it, using the cutoff.
   // START OF LOWPASS FILTER
   float* pNormalizedWaveform = batch.waveform.data_ptr<float>();

   int64_t npadded_samples = batch.waveform.size(-1);

   // We need to apply a lowpass filter to the normalized.
   auto filtered = sos_lowpass_filter(pNormalizedWaveform, npadded_samples, batch.cutoff_freq, 48000.0);

   auto pTmpTrack = pTrack->EmptyCopy();
   pTmpTrack->MakeMono();

   auto iter =
      pTmpTrack->Channels().begin();

   auto& tmpLeft = **iter++;
   tmpLeft.Append((samplePtr)filtered->data(), floatSample, npadded_samples);

   pTmpTrack->Flush();

   //apply a hard lowpass filter.
   double nyquist = 48000.0 / 2.0;
   double lowpass_ratio = batch.cutoff_freq / nyquist;

   int fs_down = int(lowpass_ratio * 48000.0);

   std::cout << "Resamping to " << fs_down << " hz." << std::endl;
   // downsample
   pTmpTrack->Resample(fs_down);

   std::cout << "Resamping back to " << 48000 << " hz." << std::endl;
   // and upsample
   pTmpTrack->Resample(48000);

   auto pChannel = pTmpTrack->GetChannel(0);

   pChannel->GetFloats(filtered->data(), 0, npadded_samples);

   auto filtered_again = sos_lowpass_filter(filtered->data(), npadded_samples, batch.cutoff_freq, 48000.0);

   return filtered_again;
}

struct OVAudioSRIntermediate
{
   //stage 1
   double input_rms;

   Batch batch;
   std::shared_ptr< std::vector<float> > lowpass_filtered;

   std::shared_ptr<AudioSR> audioSR;

   std::shared_ptr< AudioSR::AudioSRIntermediate > intermediate;

   int64_t seed;

   
};

static std::shared_ptr< OVAudioSRIntermediate > run_audiosr_stage1(float* pInput, size_t nsamples, WaveTrack::Holder pTrack, std::shared_ptr<AudioSR> audioSR, int64_t seed)
{
   try
   {
      auto intermediate = std::make_shared< OVAudioSRIntermediate >();
      intermediate->audioSR = audioSR;
      intermediate->seed = seed;
      intermediate->input_rms = calc_rms(pInput, nsamples);
      intermediate->lowpass_filtered = normalize_pad_lowpass(pInput, nsamples, pTrack, intermediate->audioSR, intermediate->batch);
      intermediate->batch.waveform_lowpass = torch::from_blob((void*)intermediate->lowpass_filtered->data(), { 1, static_cast<long>(intermediate->lowpass_filtered->size()) }, torch::kFloat);
      intermediate->intermediate = audioSR->run_audio_sr_stage1(intermediate->batch, seed);
      return intermediate;
   }
   catch (const std::exception& error) {
      std::cout << "exception in stage 1: " << error.what() << std::endl;
      return {};
   }
}

static std::future< std::shared_ptr< OVAudioSRIntermediate >> run_audiosr_stage1_async(float* pInput, size_t nsamples, WaveTrack::Holder pTrack, std::shared_ptr<AudioSR> audioSR, int64_t seed)
{
   return std::async(std::launch::async, run_audiosr_stage1, pInput, nsamples, pTrack, audioSR, seed);
}

static std::shared_ptr< OVAudioSRIntermediate > run_audiosr_stage2(std::shared_ptr< OVAudioSRIntermediate > intermediate, double unconditional_guidance_scale = 3.5, int ddim_steps = 50, std::optional< CallbackParams > callback_params = {})
{
   intermediate->intermediate = intermediate->audioSR->run_audio_sr_stage2(intermediate->intermediate, unconditional_guidance_scale, ddim_steps, intermediate->seed, callback_params);
   return intermediate;
}

static std::future< std::shared_ptr< OVAudioSRIntermediate >> run_audiosr_stage2_async(std::shared_ptr< OVAudioSRIntermediate > intermediate, double unconditional_guidance_scale = 3.5, int ddim_steps = 50, std::optional< CallbackParams > callback_params = {})
{
   return std::async(std::launch::async, run_audiosr_stage2, intermediate, unconditional_guidance_scale, ddim_steps, callback_params);
}

static torch::Tensor run_audiosr_stage3(std::shared_ptr< OVAudioSRIntermediate > intermediate)
{
   auto ret = intermediate->audioSR->run_audio_sr_stage3(intermediate->intermediate, intermediate->batch);
   return ret;
}

static std::future< torch::Tensor > run_audiosr_stage3_async(std::shared_ptr< OVAudioSRIntermediate > intermediate)
{
   return std::async(std::launch::async, run_audiosr_stage3, intermediate);
}

bool EffectOVAudioSR::Process(EffectInstance&, EffectSettings&)
{
   try
   {
      //okay, let's try to find openvino model
      //todo: Right now we're looking for the model in the 'BaseDir' (which is top-level folder of Audacity install)
      // This might be okay, but some users may not have permissions to place models there. So, also look in
      // DataDir(), which is the path to C:\Users\<user>\AppData\Roaming\audacity.
      auto model_folder_wx = wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath();
      auto model_folder = audacity::ToUTF8(wxFileName(model_folder_wx, wxT("audiosr")).GetFullPath());

      FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());

      //Note: Using a variant of wstring conversion that seems to work more reliably when there are special characters present in the path.
      std::string cache_path = wstring_to_string(wxFileName(cache_folder).GetFullPath().ToStdWstring());

      std::cout << "model_folder = " << model_folder << std::endl;
      std::cout << "cache_path = " << cache_path << std::endl;
      std::cout << "number of shifts = " << mNumberOfShifts << std::endl;

      if (m_deviceSelectionChoice >= mSupportedDevices.size())
      {
         throw std::runtime_error("Invalid device choice id:  " +
            std::to_string(m_deviceSelectionChoice));
      }

      EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };

      bool bGoodResult = true;

      std::cout << "Creating OpenVINO-based AudioSR object that will run on " << mSupportedDevices[m_deviceSelectionChoice] << std::endl;

      TotalProgress(0.01, XO("Compiling AI Model..."));

#if 1
      {
         auto device = mSupportedDevices[m_deviceSelectionChoice];
         AudioSR_Config config;
         config.first_stage_encoder_device = "GPU";
         config.vae_feature_extract_device = "GPU";
         config.ddpm__device = device;
         config.vocoder_device = "GPU";
         config.model_selection = m_modelSelectionChoice == 0 ? AudioSRModel::BASIC : AudioSRModel::SPEECH;

         bool bNeedsInit = true;
         if (_audioSR)
         {
            bNeedsInit = false;

            auto current_config = _audioSR->config();

            if (current_config.first_stage_encoder_device != config.first_stage_encoder_device ||
               current_config.vae_feature_extract_device != config.vae_feature_extract_device ||
               current_config.ddpm__device != config.ddpm__device ||
               current_config.vocoder_device != config.vocoder_device ||
               current_config.model_selection != config.model_selection
               )
            {
               bNeedsInit = true;
            }
         }

         if (bNeedsInit)
         {
            auto create_audio_sr_fut = std::async(std::launch::async, [this, &model_folder, &device, &config, &cache_path]()
            {
               // WA for OpenVINO locale caching issue (https://github.com/openvinotoolkit/openvino/issues/24370)
               OVLocaleWorkaround wa;
               if (_audioSR)
               {
                  _audioSR->set_config(config);
               }
               else
               {
                  _audioSR = std::make_shared< AudioSR >(model_folder,
                     config.first_stage_encoder_device, config.vae_feature_extract_device, config.ddpm__device, config.vocoder_device, config.model_selection, cache_path);
               }
            });

            std::future_status status;
            float total_time = 0.f;
            do {
               using namespace std::chrono_literals;
               status = create_audio_sr_fut.wait_for(0.5s);
               {
                  std::string message = "Loading Audio Super Resolution AI Model to " + device + "...";
                  if (total_time > 10)
                  {
                     message += " (This could take a while if this is the first time running this feature with this device)";
                  }
                  TotalProgress(0.01, TranslatableString{ wxString(message), {} });
               }

               total_time += 0.5;

            } while (status != std::future_status::ready);

            create_audio_sr_fut.get();
         }
      }
#endif


      std::vector< WaveTrack::Holder > tracks_to_process;

      //TODO: Hook these up to the UI
      int ddim_steps = 50;
      int64_t seed = 42;
      double unconditional_guidance_scale = 3.5;

      //Create resampled copies of the selected portion of tracks. 
      // This prevents the Resample operation to modify the user's
      // original track.
      for (auto track : outputs.Get().Selected<WaveTrack>())
      {
         auto left = track->GetChannel(0);
         auto start = left->GetStartTime();
         auto end = left->GetEndTime();
         if (track->Channels().size() > 1)
         {
            auto right = track->GetChannel(1);
            start = wxMin(start, right->GetStartTime());
            end = wxMax(end, right->GetEndTime());
         }

         // Set the current bounds to whichever left marker is
         // greater and whichever right marker is less:
         const double curT0 = std::max(start, mT0);
         const double curT1 = std::min(end, mT1);

         auto start_s = left->TimeToLongSamples(curT0);
         auto end_s = left->TimeToLongSamples(curT1);

         size_t total_samples = (end_s - start_s).as_size_t();
         Floats entire_input{ total_samples };

         // create a temporary track list to append samples to
         auto pTmpTrack = track->EmptyCopy();

         bool bOkay = left->GetFloats(entire_input.get(), start_s, total_samples);
         if (!bOkay)
         {
            throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
               std::to_string(total_samples) + " samples");
         }

         auto iter =
            pTmpTrack->Channels().begin();


         //append output samples to L & R channels.
         auto& tmpLeft = **iter++;
         tmpLeft.Append((samplePtr)entire_input.get(), floatSample, total_samples);

         if (track->Channels().size() > 1)
         {
            auto right = track->GetChannel(1);
            bOkay = right->GetFloats(entire_input.get(), start_s, total_samples);
            if (!bOkay)
            {
               throw std::runtime_error("unable to get all right samples. GetFloats() failed for " +
                  std::to_string(total_samples) + " samples");
            }

            auto& tmpRight = **iter;
            tmpRight.Append((samplePtr)entire_input.get(), floatSample, total_samples);
         }

         //flush it
         pTmpTrack->Flush();

         if (pTmpTrack->GetRate() != 48000)
         {
            pTmpTrack->Resample(48000, mProgress);
         }

         tracks_to_process.push_back(pTmpTrack);
      }

      for (size_t ti = 0; ti < tracks_to_process.size(); ti++)
      {
         auto pTrack = tracks_to_process[ti];

         std::vector< std::shared_ptr<WaveChannel> > channels_to_process;
         channels_to_process.push_back(pTrack->GetChannel(0));
         if (pTrack->Channels().size() > 1)
         {
            channels_to_process.push_back(pTrack->GetChannel(1));
         }

         auto pLeftChannel = pTrack->GetChannel(0);
         std::shared_ptr<WaveChannel> pRightChannel = {};
         if (pTrack->Channels().size() > 1)
         {
            pRightChannel = pTrack->GetChannel(1);
         }

         double trackStart = pTrack->GetStartTime();
         double trackEnd = pTrack->GetEndTime();

         // Set the current bounds to whichever left marker is
         // greater and whichever right marker is less:
         const double curT0 = trackStart;
         const double curT1 = trackEnd;

         if (curT1 > curT0) {
            auto start = pTrack->TimeToLongSamples(curT0);
            auto end = pTrack->TimeToLongSamples(curT1);

            //Get the length of the buffer (as double). len is
            //used simple to calculate a progress meter, so it is easier
            //to make it a double now than it is to do it later
            auto len = (end - start).as_double();

            size_t total_samples = (end - start).as_size_t();
            Floats entire_input{ total_samples * 2 };

            //Perform audio SR on each channel separately.

            std::vector< std::shared_ptr<std::vector<float>>> output_channels;

            int channeli = 0;
            for (auto channel : channels_to_process)
            {
               bool bOkay = channel->GetFloats(entire_input.get(), start, total_samples);
               if (!bOkay)
               {
                  throw std::runtime_error("unable to get all samples. GetFloats() failed for " +
                     std::to_string(total_samples) + " samples");
               }

               mProgressFrac = 0.0;

               if (tracks_to_process.size() > 1)
               {
                  mProgMessage = "Running Super Resolution on track " + std::to_string(ti + 1) + " / " + std::to_string(tracks_to_process.size()) + ", channel " + std::to_string(channeli++);
               }
               else if (channels_to_process.size() > 1)
               {
                  mProgMessage = "Running Super Resolution on channel " + std::to_string(channeli++);
               }
               else
               {
                  mProgMessage = "Running Super Resolution";
               }

               auto audio_sr_channel_run_future = std::async(std::launch::async,
                  [this, &entire_input, &total_samples, &ddim_steps, &seed, &unconditional_guidance_scale, &pTrack]
                  {
                     std::shared_ptr< std::vector<float> > out = std::make_shared< std::vector<float> >(total_samples);

                     // we overlap processed segments by 0.1 seconds, and then crossfade between them
                     // to suppress transition artifacts.
                     double crossfade_overlap_seconds = 0.1;

                     size_t overlap_samples = (size_t)(48000.0 * crossfade_overlap_seconds);
                     std::vector< std::pair<size_t, size_t> > segments;

                     {
                        size_t current_sample = 0;
                        size_t nchunk_samples = _audioSR->nchunk_samples();
                        while (current_sample < total_samples)
                        {
                           if (current_sample != 0)
                           {
                              current_sample -= overlap_samples;
                           }

                           std::pair<size_t, size_t> segment = { current_sample,
                             std::min(total_samples - current_sample, nchunk_samples) };

                           segments.push_back(segment);

                           current_sample += nchunk_samples;
                        }
                     }

                     CallbackParams callback_params;
                     callback_params.callback = AudioSRCallback;
                     callback_params.user = this;

                     _ddim_steps_complete = 0;
                     _total_ddim_steps = ddim_steps * segments.size();

#if 1
                     if (segments.size() > 0)
                     {
                        std::future< std::shared_ptr< OVAudioSRIntermediate >> intermediate_fut[3];
                        std::shared_ptr< OVAudioSRIntermediate > intermediate[3];

                        //fill the pipeline
                        {
                           size_t offset = segments[0].first;
                           size_t nsamples = segments[0].second;

                           float* pInput = entire_input.get() + offset;
                           intermediate_fut[0] = run_audiosr_stage1_async(pInput, nsamples, pTrack, _audioSR, seed);
                        }

                        //wait for first stage1 to complete.
                        intermediate[0] = intermediate_fut[0].get();

                        //kick off stage 2 for 0
                        intermediate_fut[0] = run_audiosr_stage2_async(intermediate[0], unconditional_guidance_scale, ddim_steps, callback_params);

                        //hack -- commenting this causes corruption for the first 10 seconds.
                        //intermediate_fut[0].wait();

                        //kick off stage 1 for 1
                        if (segments.size() > 1)
                        {
                           size_t offset = segments[1].first;
                           size_t nsamples = segments[1].second;

                           float* pInput = entire_input.get() + offset;
                           intermediate_fut[1] = run_audiosr_stage1_async(pInput, nsamples, pTrack, _audioSR, seed);
                        }

                        //wait for both to complete.
                        intermediate[0] = intermediate_fut[0].get();
                        if (intermediate_fut[1].valid())
                        {
                           intermediate[1] = intermediate_fut[1].get();
                        }

                        //loop should start here...
                        for (size_t segmenti = 0; segmenti < segments.size(); segmenti++)
                        {
                           int stage_3_index = segmenti % 3;
                           int stage_2_index = (segmenti + 1) % 3;
                           int stage_1_index = (segmenti + 2) % 3;

                           //run stage 3
                           auto stage_3_fut = run_audiosr_stage3_async(intermediate[stage_3_index]);

                           //hack
                           //stage_3_fut.wait();

                           //run stage 2
                           if (intermediate[stage_2_index])
                           {
                              intermediate_fut[stage_2_index] = run_audiosr_stage2_async(intermediate[stage_2_index], unconditional_guidance_scale, ddim_steps, callback_params);

                              //hack
                              //intermediate_fut[stage_2_index].wait();
                           }

                           //run stage 1
                           if (segments.size() > (segmenti + 2))
                           {
                              size_t offset = segments[segmenti + 2].first;
                              size_t nsamples = segments[segmenti + 2].second;

                              float* pInput = entire_input.get() + offset;
                              intermediate_fut[stage_1_index] = run_audiosr_stage1_async(pInput, nsamples, pTrack, _audioSR, seed);

                              //hack
                              //intermediate_fut[stage_1_index].wait();
                           }

                           //wait for all to complete

                           //stage 3
                           auto waveform_sr = stage_3_fut.get();

                           //stage 2
                           if (intermediate_fut[stage_2_index].valid())
                           {
                              intermediate[stage_2_index] = intermediate_fut[stage_2_index].get();
                           }

                           //stage 1
                           if (intermediate_fut[stage_1_index].valid())
                           {
                              intermediate[stage_1_index] = intermediate_fut[stage_1_index].get();
                           }

                           auto input_rms = intermediate[stage_3_index]->input_rms;

                           // clear stage3 position
                           intermediate[stage_3_index] = {};
                           intermediate_fut[stage_3_index] = {};

                           size_t offset = segments[segmenti].first;
                           size_t nsamples = segments[segmenti].second;

                           auto output_rms = calc_rms((float*)waveform_sr.data_ptr(), nsamples);

                           float ratio = input_rms / output_rms;

                           //normalize loudness back to original RMS
                           {
                              float* pSamples = (float*)waveform_sr.data_ptr();
                              for (size_t si = 0; si < nsamples; si++)
                              {
                                 pSamples[si] *= ratio;
                              }
                           }

                           //first (or only) segment. No crossfade for this one.
                           if (segmenti == 0)
                           {
                              std::memcpy(out->data() + offset, waveform_sr.data_ptr(), nsamples * sizeof(float));
                           }
                           else
                           {
                              float* pOutputCrossfadeStart = out->data() + offset;
                              float* pNewWaveform = (float*)waveform_sr.data_ptr();

                              auto crossfade_samples = std::min(overlap_samples, nsamples);
                              std::cout << "cross-fading " << crossfade_samples << " samples..." << std::endl;

                              float fade_step = 1.f / (float)(overlap_samples);
                              size_t outputi;
                              for (outputi = 0; (outputi < crossfade_samples); outputi++)
                              {
                                 float fade = pOutputCrossfadeStart[outputi] * (1 - outputi * fade_step) +
                                    pNewWaveform[outputi] * (outputi * fade_step);

                                 pOutputCrossfadeStart[outputi] = fade;
                              }

                              size_t samples_left = nsamples - outputi;
                              if (samples_left)
                              {
                                 std::memcpy(pOutputCrossfadeStart + outputi, pNewWaveform + outputi, samples_left * sizeof(float));
                              }
                           }
                        } 
                     }

#else
                     for (size_t segmenti = 0; segmenti < segments.size(); segmenti++)
                     {
                        size_t offset = segments[segmenti].first;
                        size_t nsamples = segments[segmenti].second;

#if 1
                        float* pInput = entire_input.get() + offset;
                        std::cout << "pInput should be " << (void*)(pInput) << std::endl;
                        auto stage1_fut = run_audiosr_stage1_async(pInput, nsamples, pTrack, _audioSR, seed);
                        auto intermediate = stage1_fut.get();

                        auto stage2_fut = run_audiosr_stage2_async(intermediate, unconditional_guidance_scale, ddim_steps, callback_params);
                        intermediate = stage2_fut.get();

                        auto stage3_fut = run_audiosr_stage3_async(intermediate);
                        auto waveform_sr = stage3_fut.get();
#else
                        auto intermediate = run_audiosr_stage1(entire_input.get() + offset, nsamples, pTrack, _audioSR, seed);
                        intermediate = run_audiosr_stage2(intermediate, unconditional_guidance_scale, ddim_steps, callback_params);
                        auto waveform_sr = run_audiosr_stage3(intermediate);
#endif
                        auto input_rms = intermediate->input_rms;

                        auto output_rms = calc_rms((float*)waveform_sr.data_ptr(), nsamples);

                        float ratio = input_rms / output_rms;

                        //normalize loudness back to original RMS
                        {
                           float* pSamples = (float*)waveform_sr.data_ptr();
                           for (size_t si = 0; si < nsamples; si++)
                           {
                              pSamples[si] *= ratio;
                           }
                        }

                        //first (or only) segment. No crossfade for this one.
                        if (segmenti == 0)
                        {
                           std::memcpy(out->data() + offset, waveform_sr.data_ptr(), nsamples * sizeof(float));
                        }
                        else
                        {
                           float* pOutputCrossfadeStart = out->data() + offset;
                           float* pNewWaveform = (float*)waveform_sr.data_ptr();

                           auto crossfade_samples = std::min(overlap_samples, nsamples);
                           std::cout << "cross-fading " << crossfade_samples << " samples..." << std::endl;

                           float fade_step = 1.f / (float)(overlap_samples);
                           size_t outputi;
                           for (outputi = 0; (outputi < crossfade_samples); outputi++)
                           {
                              float fade = pOutputCrossfadeStart[outputi] * (1 - outputi * fade_step) +
                                 pNewWaveform[outputi] * (outputi * fade_step);

                              pOutputCrossfadeStart[outputi] = fade;
                           }

                           size_t samples_left = nsamples - outputi;
                           if (samples_left)
                           {
                              std::memcpy(pOutputCrossfadeStart + outputi, pNewWaveform + outputi, samples_left * sizeof(float));
                           }
                        }
                     }
#endif

                     return out;
                  }
               );

               std::future_status status;

               do {
                  using namespace std::chrono_literals;
                  status = audio_sr_channel_run_future.wait_for(0.1s);
                  {
                     std::lock_guard<std::mutex> guard(mProgMutex);
                     mProgress->SetMessage(TranslatableString{ wxString(mProgMessage), {} });
                     if (TotalProgress(mProgressFrac))
                     {
                        mIsCancelled = true;
                     }
                  }

               } while (status != std::future_status::ready);

               auto out = audio_sr_channel_run_future.get();

               output_channels.push_back(out);
            }

            auto pProject = FindProject();
            const auto& selectedRegion =
               ViewInfo::Get(*pProject).selectedRegion;

            auto orig_track_name = pTrack->GetName();
            // Workaround for 3.4.X issue where setting name of a new output track
            // retains the label of the track that it was copied from. So, we'll
            // change the name of the input track here, copy it, and then change it
            // back later.
            pTrack->SetName(orig_track_name + wxString("-AudioSR"));

            //Create new output track from input track.
            auto newOutputTrack = pTrack->EmptyCopy();

            // create a temporary track list to append samples to
            auto pTmpTrack = pTrack->EmptyCopy();
            auto iter = pTmpTrack->Channels().begin();

            if (output_channels.size() > 0)
            {
               auto left_channel_out = output_channels[0];
               auto& tmpLeft = **iter++;
               tmpLeft.Append((samplePtr)left_channel_out->data(), floatSample, total_samples);
            }

            if (output_channels.size() > 1)
            {
               auto right_channel_out = output_channels[1];
               auto& tmpRight = **iter;
               tmpRight.Append((samplePtr)right_channel_out->data(), floatSample, total_samples);
            }

            //flush it
            pTmpTrack->Flush();

            // Clear & Paste into new output track
            newOutputTrack->ClearAndPaste(selectedRegion.t0() - pTmpTrack->GetStartTime(),
               selectedRegion.t1() - pTmpTrack->GetStartTime(), *pTmpTrack);

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

            // Add the new track list to the output.
            outputs.AddToOutputTracks(std::move(newOutputTrack));

            // Change name back to original. Part of workaround described above.
            pTrack->SetName(orig_track_name);

         }
         else
         {
            throw std::runtime_error("unexpected case encountered where curT0 (" + std::to_string(curT0) +
               ") <= curT1(" + std::to_string(curT1) + ")");
         }
      }

      if (bGoodResult)
         outputs.Commit();

      return bGoodResult;
   }
   catch (const std::exception& error) {
      wxLogError("In Super Resolution, exception: %s", error.what());
      EffectUIServices::DoMessageBox(*this,
         XO("Super Resolution failed. See details in Help->Diagnostics->Show Log..."),
         wxICON_STOP,
         XO("Error"));
   }

   return false;
}

bool EffectOVAudioSR::TransferDataToWindow(const EffectSettings&)
{
   if (!mUIParent || !mUIParent->TransferDataToWindow())
   {
      return false;
   }

   EffectEditor::EnablePreview(mUIParent, false);

   return true;
}
