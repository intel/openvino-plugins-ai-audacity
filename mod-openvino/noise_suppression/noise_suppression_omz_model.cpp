// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#include "noise_suppression_omz_model.h"


NoiseSuppressionOMZModel::NoiseSuppressionOMZModel(std::string model_path, std::string device, std::string cache_dir)
{
   _compile_noise_suppression_model(model_path, device, cache_dir);
}

bool NoiseSuppressionOMZModel::run(std::shared_ptr<WaveChannel> pChannel, sampleCount start, size_t total_samples, ProgressCallbackFunc callback, void* callback_user)
{
   bool ret = true;

   auto infer_request = _compiledModel.create_infer_request();
   auto inputs = _compiledModel.inputs();
   auto outputs = _compiledModel.outputs();

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


   // Prepare input
   // get size of network input (patch_size)
   std::string input_name("input");
   ov::Shape inp_shape = _compiledModel.input(input_name).get_shape();
   size_t patch_size = inp_shape[1];

   // try to get delay output and freq output for model
   int delay = 0;
   infer_request.infer();
   for (size_t i = 0; i < outputs.size(); i++) {
      std::string out_name = outputs[i].get_any_name();
      if (out_name == "delay") {
         delay = infer_request.get_tensor("delay").data<int>()[0];
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
            ov::Shape state_shape = _compiledModel.input(inp_state_name).get_shape();
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
      if (callback)
      {
         if (!callback((double)(i + 1) / double(iter), callback_user))
         {
            return false;
         }
      }
   } // for iter

   ret = pChannel->Set((samplePtr)entire_output.get(), floatSample, start, total_samples);
   if (!ret)
   {
      throw std::runtime_error("WaveTrack::Set failed for " + std::to_string(total_samples) + " samples.");
   }

   return ret;
}

void NoiseSuppressionOMZModel::_compile_noise_suppression_model(std::string model_path, std::string device, std::string cache_dir)
{
   _core = std::make_shared< ov::Core >();

   _core->set_property(ov::cache_dir(cache_dir));

   std::shared_ptr<ov::Model> model = _core->read_model(model_path);

   ov::OutputVector inputs = model->inputs();
   ov::OutputVector outputs = model->outputs();

   _compiledModel = _core->compile_model(model, device, {});

   auto infer_request = _compiledModel.create_infer_request();
   infer_request.infer();

   _freq_model = 16000;

   for (size_t i = 0; i < outputs.size(); i++) {
      std::string out_name = outputs[i].get_any_name();
      if (out_name == "freq") {
         _freq_model = infer_request.get_tensor("freq").data<int>()[0];
      }
   }
}


