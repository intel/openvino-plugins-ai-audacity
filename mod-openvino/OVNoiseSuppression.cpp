// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "OVNoiseSuppression.h"
#include "WaveTrack.h"
#include "EffectOutputTracks.h"
#include "effects/EffectEditor.h"
#include <math.h>
#include <iostream>

#include <wx/log.h>

#include "ViewInfo.h"
#include "TimeWarper.h"

#include <wx/intl.h>
#include <wx/valgen.h>

#include "ShuttleGui.h"

#include <wx/choice.h>
#include "FileNames.h"
#include "CodeConversions.h"

#include "LoadEffects.h"
#include <future>

#include <openvino/openvino.hpp>

const ComponentInterfaceSymbol EffectOVNoiseSuppression::Symbol{ XO("OpenVINO Noise Suppression") };

namespace { BuiltinEffectsModule::Registration< EffectOVNoiseSuppression > reg; }


EffectOVNoiseSuppression::EffectOVNoiseSuppression()
{
   ov::Core core;

   auto ov_supported_device = core.get_available_devices();
   for (auto d : ov_supported_device)
   {
      //GNA devices are not supported
      if (d.find("GNA") != std::string::npos) continue;

      mSupportedDevices.push_back(d);
      mGuiDeviceSelections.push_back({ TranslatableString{ wxString(d), {}} });

   }
   
   mSupportedModels = { "noise-suppression-denseunet-ll-0001" };

   for (auto m : mSupportedModels)
   {
      mGuiModelSelections.push_back({ TranslatableString{ wxString(m), {}} });
   }

}

EffectOVNoiseSuppression::~EffectOVNoiseSuppression()
{

}

// ComponentInterface implementation
ComponentInterfaceSymbol EffectOVNoiseSuppression::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVNoiseSuppression::GetDescription() const
{
   return XO("Applies AI Background Noise Suppression using OpenVINO");
}

VendorSymbol EffectOVNoiseSuppression::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

// EffectDefinitionInterface implementation

EffectType EffectOVNoiseSuppression::GetType() const
{
   return EffectTypeProcess;
}

bool EffectOVNoiseSuppression::IsInteractive() const
{
   return true;
}

std::unique_ptr<EffectEditor> EffectOVNoiseSuppression::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess&,
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
            .AddChoice(XXO("Noise Suppression Model:"),
               Msgids(mGuiModelSelections.data(), mGuiModelSelections.size()));
      }
      S.EndMultiColumn();
   }
   S.EndVerticalLay();

   return nullptr;
}

void EffectOVNoiseSuppression::CompileNoiseSuppression(ov::CompiledModel& compiledModel)
{
   FilePath model_folder = FileNames::MkDir(wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath());

   auto model_file = audacity::ToUTF8(mTypeChoiceModelCtrl->GetString(m_modelSelectionChoice)) + ".xml";

   std::string  model_path = audacity::ToUTF8(wxFileName(model_folder, wxString(model_file))
         .GetFullPath());

   std::cout << "Using model path = " << model_path << std::endl;

   FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());
   std::string cache_path = audacity::ToUTF8(wxFileName(cache_folder).GetFullPath());

   ov::Core core;

   core.set_property(ov::cache_dir(cache_path));

   std::shared_ptr<ov::Model> model = core.read_model(model_path);

   ov::OutputVector inputs = model->inputs();
   ov::OutputVector outputs = model->outputs();

   // get state names pairs (inp,out) and compute overall states size
   size_t state_size = 0;
   std::vector<std::pair<std::string, std::string>> state_names;
   for (size_t i = 0; i < inputs.size(); i++) {
      std::string inp_state_name = inputs[i].get_any_name();
      if (inp_state_name.find("inp_state_") == std::string::npos)
         continue;

      std::string out_state_name(inp_state_name);
      out_state_name.replace(0, 3, "out");

      // find corresponding output state
      if (outputs.end() == std::find_if(outputs.begin(), outputs.end(), [&out_state_name](const ov::Output<ov::Node>& output) {
         return output.get_any_name() == out_state_name;
         }))
         throw std::runtime_error("model output state name does not correspond input state name");

         state_names.emplace_back(inp_state_name, out_state_name);

         ov::Shape shape = inputs[i].get_shape();
         size_t tensor_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

         state_size += tensor_size;
   }

   if (state_size == 0)
      throw std::runtime_error("no expected model state inputs found");

   compiledModel = core.compile_model(model, mSupportedDevices[m_deviceSelectionChoice], {});
   
}

bool EffectOVNoiseSuppression::ApplyNoiseSuppression(std::shared_ptr<WaveChannel> pChannel, ov::CompiledModel& compiledModel, sampleCount start, size_t total_samples)
{
   bool ret = true;
   try
   {
      auto inputs = compiledModel.inputs();
      auto outputs = compiledModel.outputs();

      // get state names pairs (inp,out) and compute overall states size
      size_t state_size = 0;
      std::vector<std::pair<std::string, std::string>> state_names;
      for (size_t i = 0; i < inputs.size(); i++) {
         std::string inp_state_name = inputs[i].get_any_name();
         if (inp_state_name.find("inp_state_") == std::string::npos)
            continue;

         std::string out_state_name(inp_state_name);
         out_state_name.replace(0, 3, "out");

         // find corresponding output state
         if (outputs.end() == std::find_if(outputs.begin(), outputs.end(), [&out_state_name](const ov::Output<const ov::Node>& output) {
            return output.get_any_name() == out_state_name;
            }))
            throw std::runtime_error("model output state name does not correspond input state name");

            state_names.emplace_back(inp_state_name, out_state_name);

            ov::Shape shape = inputs[i].get_shape();
            size_t tensor_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

            state_size += tensor_size;
      }

      if (state_size == 0)
         throw std::runtime_error("no expected model state inputs found");

      ov::InferRequest infer_request = compiledModel.create_infer_request();

      // Prepare input
      // get size of network input (patch_size)
      std::string input_name("input");
      ov::Shape inp_shape = compiledModel.input(input_name).get_shape();
      size_t patch_size = inp_shape[1];

      // try to get delay output and freq output for model
      int delay = 0;
      int freq_model = 16000; // default sampling rate for model
      infer_request.infer();
      for (size_t i = 0; i < outputs.size(); i++) {
         std::string out_name = outputs[i].get_any_name();
         if (out_name == "delay") {
            delay = infer_request.get_tensor("delay").data<int>()[0];
         }
         if (out_name == "freq") {
            freq_model = infer_request.get_tensor("freq").data<int>()[0];
         }
      }

      std::cout << "delay = " << delay << std::endl;
      size_t iter = 1 + ((total_samples + delay) / patch_size);
      size_t inp_size = patch_size * iter;

      Floats entire_input{ inp_size };
      bool bOkay = pChannel->GetFloats(entire_input.get(), start, total_samples);
      if (!bOkay)
      {
         throw std::runtime_error("Unable to get " + std::to_string(total_samples) + " samples.");
      }

      //zero out the stuff we buffered
      for (int i = total_samples; i < inp_size; i++)
      {
         entire_input[i] = 0.f;
      }

      Floats entire_output{ inp_size };

      float* pInput = entire_input.get();
      float* pOutput = entire_output.get();

      for (size_t i = 0; i < iter; ++i) {
         ov::Tensor input_tensor(ov::element::f32, inp_shape, &pInput[i * patch_size]);
         infer_request.set_tensor(input_name, input_tensor);

         for (auto& state_name : state_names) {
            const std::string& inp_state_name = state_name.first;
            const std::string& out_state_name = state_name.second;
            ov::Tensor state_tensor;
            if (i > 0) {
               // set input state by coresponding output state from prev infer
               state_tensor = infer_request.get_tensor(out_state_name);
            }
            else {
               // first iteration. set input state to zero tensor.
               ov::Shape state_shape = compiledModel.input(inp_state_name).get_shape();
               state_tensor = ov::Tensor(ov::element::f32, state_shape);
               std::memset(state_tensor.data<float>(), 0, state_tensor.get_byte_size());
            }
            infer_request.set_tensor(inp_state_name, state_tensor);
         }

         infer_request.infer();

         {
            // process output
            float* src = infer_request.get_tensor("output").data<float>();
            float* dst = &pOutput[i * patch_size];
            std::memcpy(dst, src, patch_size * sizeof(float));
         }

         // This returns true if the user clicks 'cancel' button
         if (TotalProgress((double)(i + 1) / double(iter)))
         {
            return false;
         }

      } // for iter

      ret = pChannel->Set((samplePtr)entire_output.get(), floatSample, start, total_samples);
      if (!ret)
      {
         throw std::runtime_error("WaveTrack::Set failed for " + std::to_string(total_samples) + " samples.");
      }

   }
   catch (const std::exception& error) {
      wxLogError("In Noise Suppression, exception: %s", error.what());
      EffectUIServices::DoMessageBox(*this,
         XO("Noise Suppression failed. See details in Help->Diagnostics->Show Log..."),
         wxICON_STOP,
         XO("Error"));
      return false;
   }

   return ret;
}

bool EffectOVNoiseSuppression::Process(EffectInstance&, EffectSettings&)
{
   EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
   bool bGoodResult = true;

   ov::CompiledModel compiledModel;
   try
   {
      auto compile_compiledModel_fut = std::async(std::launch::async, [this, &compiledModel]() {
         try {
            CompileNoiseSuppression(compiledModel);
            return true;
         }
         catch (const std::exception& error) {
            wxLogError("In Noise Suppression Compilation, exception: %s", error.what());
            EffectUIServices::DoMessageBox(*this,
               XO("Noise Suppression failed. See details in Help->Diagnostics->Show Log..."),
               wxICON_STOP,
               XO("Error"));
            return false;
         }
         });

      std::future_status status;
      float total_time = 0.f;
      do {
         using namespace std::chrono_literals;
         status = compile_compiledModel_fut.wait_for(0.5s);
         {
            std::string message = "Compiling Noise Suppression AI Model for " + mSupportedDevices[m_deviceSelectionChoice] + "...";
            if (total_time > 10)
            {
               message += " (This could take a while if this is the first time running this feature with this device)";
            }
            TotalProgress(0.01, TranslatableString{ wxString(message), {} });
         }

         total_time += 0.5;

      } while (status != std::future_status::ready);

      auto success = compile_compiledModel_fut.get();

      if (!success)
      {
         std::cout << "CompileNoiseSuppression routine failed." << std::endl;
         return false;
      }
   }
   catch (const std::exception& error) {
      std::cout << "CompileNoiseSuppression routine failed: " << error.what() << std::endl;
      return false;
   }

   //mCurTrackNum = 0;
   size_t wavetracki = 0;
   for (auto pOutWaveTrack : outputs.Get().Selected<WaveTrack>())
   {
      //Get start and end times from track
      double trackStart = pOutWaveTrack->GetStartTime();
      double trackEnd = pOutWaveTrack->GetEndTime();

      // Set the current bounds to whichever left marker is
      // greater and whichever right marker is less:
      const double curT0 = std::max(trackStart, mT0);
      const double curT1 = std::min(trackEnd, mT1);

      // Process only if the right marker is to the right of the left marker
      if (curT1 > curT0) {
         double origRate = pOutWaveTrack->GetRate();
         pOutWaveTrack->Resample(16000);

         //Transform the marker timepoints to samples
         auto start = pOutWaveTrack->TimeToLongSamples(curT0);
         auto end = pOutWaveTrack->TimeToLongSamples(curT1);

         size_t total_samples = (end - start).as_size_t();

         for (size_t channeli = 0; channeli < pOutWaveTrack->Channels().size(); channeli++)
         {
            std::string message = "Running Noise Suppression on Track " + std::to_string(wavetracki) + ", channel " + std::to_string(channeli);
            if (TotalProgress(0.01, TranslatableString{ wxString(message), {} }))
            {
               return false;
            }

            auto pChannel = pOutWaveTrack->GetChannel(channeli);

            if (!ApplyNoiseSuppression(pChannel, compiledModel, start, total_samples))
            {
               return false;
            }

         }

         //resample back to original rate.
         pOutWaveTrack->Resample(origRate);
      }

      wavetracki++;
   }

   if (bGoodResult)
      outputs.Commit();

   return bGoodResult;
}

