// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <future>
#include "musicgen_decoder_model.h"
#include "musicgen_utils.h"
#include "musicgen_config.h"

namespace ov_musicgen
{
   class MusicgenDecoderModelFullStaticBatch1 : public MusicgenDecoderModel
   {
   public:

      const size_t N_LAYERS = 24;
      const size_t MAX_TENSOR_HEIGHT = 1503;

      MusicgenDecoderModelFullStaticBatch1(ov::Core& core, MusicGenConfig& config)
      {
         auto model_folder = config.model_folder;
         std::vector<std::string> devices = { config.musicgen_decode_device0, config.musicgen_decode_device1 };

         //if device0 or device1 are set to "CPU", use CPU for the initial device.
         // Otherwise, ideally, we use GPU for that -- so try to find a supported GPU
         // and set it to that.
         std::string initial_device;
         if (config.musicgen_decode_device0 == "CPU" || config.musicgen_decode_device1 == "CPU")
         {
            initial_device = "CPU";
         }
         else
         {
            //iterate through supported device list, looking for a GPU device.
            std::optional< std::string > gpu_device;
            {
               auto device_list = core.get_available_devices();

               for (auto d : device_list)
               {
                  if (d.find("GPU") != std::string::npos)
                  {
                     gpu_device = d;
                     break;
                  }
               }
            }

            if (gpu_device)
            {
               initial_device = *gpu_device;
            }
            else
            {
               initial_device = "CPU";
            }
         }

         std::cout << "Using initial device as: " << initial_device << std::endl;

         auto initial_tensortype = initial_device == "CPU" ? ov::element::f32 : ov::element::f16;

         if (config.bStereo)
         {
            model_folder = FullPath(model_folder, "stereo");
         }
         else
         {
            model_folder = FullPath(model_folder, "mono");
         }

         using namespace std::chrono;
         using Clock = std::chrono::high_resolution_clock;

         //prep decoder model

         //prep decoder model
         {
            //for each batch stuff
            for (int dbi = 0; dbi < 2; dbi++)
            {
               auto tensortype = devices[dbi] == "CPU" ? ov::element::f32 : ov::element::f16;

               std::string decoder_model_path, binfile;
               switch (config.model_selection)
               {
               case MusicGenConfig::ModelSelection::MUSICGEN_SMALL_FP16:
                  decoder_model_path = FullPath(model_folder, "musicgen_decoder_static_batch1.xml");
                  binfile = FullPath(model_folder, "musicgen_decoder_combined_weights.bin");
                  break;

               case  MusicGenConfig::ModelSelection::MUSICGEN_SMALL_INT8:
                  decoder_model_path = FullPath(model_folder, "musicgen_decoder_static_batch1_int8.xml");
                  binfile = FullPath(model_folder, "musicgen_decoder_combined_weights_int8.bin");
                  break;

               default:
                  throw std::runtime_error("Invalid model selection");
                  break;
               }

               std::cout << " Using model=" << decoder_model_path << ", " << binfile << std::endl;

               std::shared_ptr<ov::Model> model = core.read_model(decoder_model_path, binfile);

               ov::preprocess::PrePostProcessor ppp(model);

               for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
               {
                  for (size_t i = 0; i < 4; i++)
                  {
                     std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                     ppp.input(tensorname).tensor().set_element_type(tensortype);
                  }

                  for (int i = 0; i < 2; i++)
                  {
                     std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                     ppp.output(tensorname).tensor().set_element_type(tensortype);

                  }
               }

               model = ppp.build();


               //logBasicModelInfo(model);

               ov::CompiledModel compiledModel;

               uint64_t  t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
               if (dbi == 0)
               {
                  compiledModel = core.compile_model(model, devices[0]);
               }
               else
               {
                  if (devices[1] == devices[0])
                  {
                     compiledModel = _decoder_batch[0]._infer_request.get_compiled_model();
                  }
                  else
                  {
                     compiledModel = core.compile_model(model, devices[1]);
                  }
               }
               uint64_t  t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
               std::cout << "    compile time for device=" << devices[dbi] << " = " << (t1 - t0) << " ms" << std::endl;

               //first inference is usually a bit longer due to some lazy initialization, so trigger a warm up inference.
               _decoder_batch[dbi]._infer_request = compiledModel.create_infer_request();
               _decoder_batch[dbi]._infer_request.infer();

               //std::cout << "hidden states strides = " << _decoder_batch[dbi]._infer_request.get_tensor("input_hidden_states").get_strides() << std::endl;
               //std::cout << "encoder_attention_mask strides = " << _decoder_batch[dbi]._infer_request.get_tensor("encoder_attention_mask").get_strides() << std::endl;
               //std::cout << "custom_attention_mask strides = " << _decoder_batch[dbi]._infer_request.get_tensor("custom_attention_mask").get_strides() << std::endl;


               _decoder_batch[dbi]._hidden_states = wrap_ov_tensor_as_torch(_decoder_batch[dbi]._infer_request.get_tensor("input_hidden_states"));
               _decoder_batch[dbi]._encoder_attention_mask_ov = _decoder_batch[dbi]._infer_request.get_tensor("encoder_attention_mask");
               _decoder_batch[dbi]._encoder_attention_mask = wrap_ov_tensor_as_torch(_decoder_batch[dbi]._encoder_attention_mask_ov);

               _decoder_batch[dbi]._custom_attention_mask = _decoder_batch[dbi]._infer_request.get_tensor("custom_attention_mask");

               _decoder_batch[dbi]._past_key_values_ov.resize(N_LAYERS);
               _decoder_batch[dbi]._past_key_values_torch.resize(N_LAYERS);
               for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
               {
                  _decoder_batch[dbi]._past_key_values_ov[layeri].resize(4);
                  _decoder_batch[dbi]._past_key_values_torch[layeri].resize(4);

                  for (size_t i = 0; i < 4; i++)
                  {
                     std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

                     _decoder_batch[dbi]._past_key_values_ov[layeri][i] = _decoder_batch[dbi]._infer_request.get_tensor(tensorname);

                     //wrap ov tensor as torch tensor
                     _decoder_batch[dbi]._past_key_values_torch[layeri][i] = wrap_ov_tensor_as_torch(_decoder_batch[dbi]._past_key_values_ov[layeri][i]);
                  }
               }

               //get last hidden state tensor
               _decoder_batch[dbi]._last_hidden_state_ov = _decoder_batch[dbi]._infer_request.get_tensor("last_hidden_state");

               //std::cout << "_last_hidden_state_ov strides = " << _decoder_batch[dbi]._last_hidden_state_ov.get_strides() << std::endl;

               //wrap it as a torch::Tensor
               _decoder_batch[dbi]._last_hidden_state = wrap_ov_tensor_as_torch(_decoder_batch[dbi]._last_hidden_state_ov);

               _decoder_batch[dbi]._new_key_values_ov.resize(N_LAYERS);
               _decoder_batch[dbi]._new_key_values.resize(N_LAYERS);
               for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
               {
                  _decoder_batch[dbi]._new_key_values[layeri].resize(2);
                  _decoder_batch[dbi]._new_key_values_ov[layeri].resize(2);

                  for (int i = 0; i < 2; i++)
                  {
                     std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

                     auto ov_tensor = _decoder_batch[dbi]._infer_request.get_tensor(tensorname);

                     _decoder_batch[dbi]._new_key_values_ov[layeri][i] = ov_tensor;

                     //wrap ov tensor as torch tensor
                     _decoder_batch[dbi]._new_key_values[layeri][i] = wrap_ov_tensor_as_torch(ov_tensor);
                  }
               }

               std::cout << "Batch1 Decoder, Index " << dbi << ":" << std::endl;
               std::cout << "    tensortype = " << tensortype << std::endl;
               std::cout << "    max token length = " << MaxNewTokens() << std::endl;
            }

            auto last_hidden_shape = _decoder_batch[0]._last_hidden_state_ov.get_shape();
            last_hidden_shape[0] = 2;
            _last_hidden_state_ov = ov::Tensor(_decoder_batch[0]._last_hidden_state_ov.get_element_type(), last_hidden_shape);
            _last_hidden_state = wrap_ov_tensor_as_torch(_last_hidden_state_ov);


         }

         //prep initial model
         {
            auto model_path = FullPath(model_folder, "initial_cross_attn_kv_producer.xml");
            std::shared_ptr<ov::Model> model = core.read_model(model_path);

            model->reshape({ {2, _decoder_batch[0]._past_key_values_ov[0][2].get_shape()[2], 1024} });

            ov::preprocess::PrePostProcessor ppp(model);

            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (int i = 0; i < 2; i++)
               {
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                  ppp.output(tensorname).tensor().set_element_type(initial_tensortype);
               }
            }

            model = ppp.build();

            //make this one configurable? 
            ov::CompiledModel compiledModel = core.compile_model(model, initial_device);

            _infer_request_initial = compiledModel.create_infer_request();
            _infer_request_initial.infer();

            _intiial_encoder_hidden_state = wrap_ov_tensor_as_torch(_infer_request_initial.get_input_tensor());
            _initial_past_key_values.resize(N_LAYERS);
            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               _initial_past_key_values[layeri].resize(2);
               for (int i = 0; i < 2; i++)
               {
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

                  // note, the output from the initial model is copied at runtime to the input past key vals.
               }
            }
         }

         //todo: this is only needed for song continuation, so make it possible to construct without it, if we're not doing that.
         {

            std::string model_path;
            switch (config.m_continuation_context)
            {
            case MusicGenConfig::ContinuationContext::FIVE_SECONDS:
            {
               model_path = FullPath(model_folder, "musicgen_decoder_static0_5s.xml");
               auto attn_mask_raw_file = FullPath(config.model_folder, "attention_mask_from_prepare_4d_causal_5s.raw");
               _attention_mask = read_tensor(attn_mask_raw_file, { 2, 1, 251, 251 });
            }
            break;

            case MusicGenConfig::ContinuationContext::TEN_SECONDS:
            {
               model_path = FullPath(model_folder, "musicgen_decoder_static0_10s.xml");
               auto attn_mask_raw_file = FullPath(config.model_folder, "attention_mask_from_prepare_4d_causal_10s.raw");
               _attention_mask = read_tensor(attn_mask_raw_file, { 2, 1, 501, 501 });
            }
            break;
            }

            auto binfile = FullPath(model_folder, "musicgen_decoder_combined_weights.bin");
            std::cout << "reading model as " << model_path << ", " << binfile << std::endl;

            std::shared_ptr<ov::Model> model = core.read_model(model_path, binfile);

            ov::preprocess::PrePostProcessor ppp(model);

            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (size_t i = 0; i < 2; i++)
               {
                  std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i + 2);
                  ppp.input(tensorname).tensor().set_element_type(initial_tensortype);
               }

               for (int i = 0; i < 2; i++)
               {
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                  ppp.output(tensorname).tensor().set_element_type(initial_tensortype);

               }
            }

            model = ppp.build();

            std::cout << "large context model" << std::endl;
            logBasicModelInfo(model);

            ov::CompiledModel compiledModel = core.compile_model(model, initial_device);

            _infer_request_large_context = compiledModel.create_infer_request();
            _infer_request_large_context.infer();

            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (int i = 0; i < 2; i++)
               {
                  std::string intial_tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                  //link the initial model with the this model
                  auto ov_tens = _infer_request_initial.get_tensor(intial_tensorname);

                  std::string large_tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i + 2);
                  _infer_request_large_context.set_tensor(large_tensorname, ov_tens);
               }
            }
         }


         //std::cout << "Resetting!" << std::endl;
         Reset();

         std::cout << "construction complete!" << std::endl;
      }

      virtual void Reset() override
      {
         _past_length = 0;

         for (int dbi = 0; dbi < 2; dbi++)
         {
            memset(_decoder_batch[dbi]._custom_attention_mask.data<float>(), 0, _decoder_batch[dbi]._custom_attention_mask.get_byte_size());
            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (size_t i = 0; i < 4; i++)
               {
                  memset(_decoder_batch[dbi]._past_key_values_ov[layeri][i].data(), 0, _decoder_batch[dbi]._past_key_values_ov[layeri][i].get_byte_size());
               }
            }

            float* pAttnMask = _decoder_batch[dbi]._custom_attention_mask.data<float>();
            //std::cout << "_custom_attention_mask.get_size() = " << _custom_attention_mask.get_size() << std::endl;
            for (int i = 0; i < _decoder_batch[dbi]._custom_attention_mask.get_size(); i++)
            {
               pAttnMask[i] = -INFINITY;
            }

            {
               float* pAttnMask = _decoder_batch[dbi]._encoder_attention_mask_ov.data<float>();
               for (int i = 0; i < _decoder_batch[dbi]._encoder_attention_mask_ov.get_size(); i++)
               {
                  pAttnMask[i] = -INFINITY;
               }
            }
         }

         memset(_infer_request_initial.get_input_tensor().data<float>(), 0, _infer_request_initial.get_input_tensor().get_byte_size());
      }

      virtual void ShiftLeft(int64_t ntokens) override
      {

      }

      virtual int64_t MaxNewTokens() override
      {
         return _decoder_batch[0]._past_key_values_ov[0][0].get_shape()[2];
      }

      virtual ov::Tensor run(torch::Tensor hidden_states, std::optional<torch::Tensor> encoder_hidden_states, std::optional<torch::Tensor> encoder_attention_mask) override
      {
         ITT_SCOPED_TASK(MusicgenModelStatic_run)
            ov::Tensor last_hidden_states_ret;
         if (_past_length == 0)
         {
            ITT_SCOPED_TASK(initial_infer)
               using namespace torch::indexing;
            _intiial_encoder_hidden_state.index_put_({ Slice(), Slice(0, encoder_hidden_states->sizes()[1]), Slice() }, *encoder_hidden_states);

            _infer_request_initial.infer();

            //copy the result of this inference to our 2 batch-1 infer_requests.
            //TODO: Handle potential precision difference.
            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (int i = 0; i < 2; i++)
               {
                  using namespace torch::indexing;
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                  auto new_key_val_torch = wrap_ov_tensor_as_torch(_infer_request_initial.get_tensor(tensorname));

                  for (int64_t dbi = 0; dbi < 2; dbi++)
                  {
                     auto dtype = _decoder_batch[dbi]._past_key_values_torch[layeri][i + 2].dtype().toScalarType();

                     _decoder_batch[dbi]._past_key_values_torch[layeri][i + 2].copy_(
                        new_key_val_torch.index({ Slice(dbi, dbi + 1), Slice(), Slice(), Slice() }).toType(dtype)
                     );
                  }
               }
            }
         }

         if (hidden_states.sizes()[1] > 1)
         {
            ITT_SCOPED_TASK(large_context_path)
               using namespace torch::indexing;
            std::cout << "large context path!" << std::endl;

            auto input_hidden_states = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor("input_hidden_states"));
            auto input_attention_mask = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor("attention_mask"));


            auto input_encoder_attention_mask = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor("encoder_attention_mask"));

            //std::cout << "Sizes: " << input_hidden_states.sizes() << " " << hidden_states.sizes() << std::endl;
            //std::cout << "Sizes: " << input_attention_mask.sizes() << " " << _attention_mask.sizes() << std::endl;
            //std::cout << "Sizes: " << input_encoder_attention_mask.sizes() << " " << encoder_attention_mask.sizes() << std::endl;
            input_hidden_states.copy_(hidden_states);
            input_attention_mask.copy_(_attention_mask);

            //first, fill with -INF
            input_encoder_attention_mask.copy_(torch::full(input_encoder_attention_mask.sizes(), -INFINITY));

            //then slice the valid values in.
            if (encoder_attention_mask)
            {
               input_encoder_attention_mask.index_put_({ Slice(), Slice(), Slice(), Slice(0, encoder_attention_mask->sizes()[3]) }, *encoder_attention_mask);
            }

            {
               ITT_SCOPED_TASK(infer)
                  _infer_request_large_context.infer();
            }

            auto ov_last_hidden_state = _infer_request_large_context.get_tensor("last_hidden_state");

            last_hidden_states_ret = ov_last_hidden_state;

            //save_tensor_to_disk(ov_last_hidden_state, "ov_last_hidden_state.raw");

            ITT_SCOPED_TASK(update_past_key_values);

            int64_t valid_height = ov_last_hidden_state.get_shape()[1];

            //insert the new key / value tensors into the past keys tensor
            for (int64_t dbi = 0; dbi < 2; dbi++)
            {
               for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
               {
                  for (int i = 0; i < 2; i++)
                  {
                     using namespace torch::indexing;
                     std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

                     auto new_key_val_torch = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor(tensorname));
                     _decoder_batch[dbi]._past_key_values_torch[layeri][i].index_put_({ Slice(), Slice(), Slice(0, valid_height), Slice() },
                        new_key_val_torch.index({ Slice(dbi, dbi + 1), Slice(), Slice(), Slice() }));
                  }
               }

               float* pAttnMask = _decoder_batch[dbi]._custom_attention_mask.data<float>();
               memset(pAttnMask, 0, valid_height * sizeof(float));

            }

            _past_length = valid_height;
         }
         else
         {
            //set attention mask
            for (int dbi = 0; dbi < 2; dbi++)
            {
               float* pAttnMask = _decoder_batch[dbi]._custom_attention_mask.data<float>();
               if (_past_length > 0)
               {
                  pAttnMask[_past_length - 1] = 0.f;
               }

               pAttnMask[_decoder_batch[dbi]._custom_attention_mask.get_size() - 1] = 0.f;
            }

            bool bDebug = false;
            static int dd = 0;
            if (dd++ == 0)
               bDebug = true;

            //set input tensors
            //set input tensors
            //dump_tensor(encoder_attention_mask, "encoder_attention_mask.raw");


            for (int dbi = 0; dbi < 2; dbi++)
            {
               using namespace torch::indexing;
               _decoder_batch[dbi]._hidden_states.index_put_({ Slice(), Slice(), Slice() }, hidden_states.index({ Slice(dbi, dbi + 1), Slice(), Slice() }));

               if (encoder_attention_mask)
               {
                  _decoder_batch[dbi]._encoder_attention_mask.index_put_({ Slice(), Slice(), Slice(), Slice(0, encoder_attention_mask->sizes()[3]) }, encoder_attention_mask->index({ Slice(dbi, dbi + 1), Slice(), Slice(), Slice() }));
               }
            }

            // dump_tensor(_decoder_batch[0]._encoder_attention_mask, "encoder_attention_mask0.raw");
             //dump_tensor(_decoder_batch[1]._encoder_attention_mask, "encoder_attention_mask1.raw");

            if (0)
            {
               ITT_SCOPED_TASK(infers_async)
                  _decoder_batch[0]._infer_request.start_async();
               _decoder_batch[1]._infer_request.start_async();

               _decoder_batch[0]._infer_request.wait();
               _decoder_batch[1]._infer_request.wait();
            }
            else if (0)
            {
               _decoder_batch[0]._infer_request.infer();
               _decoder_batch[1]._infer_request.infer();
            }
            else
            {
               auto run_first_batch_fut = std::async(std::launch::async, [this]() {
                  ITT_SCOPED_TASK(infer0)
                     _decoder_batch[0]._infer_request.infer();
                  });

               auto run_second_batch_fut = std::async(std::launch::async, [this]() {
                  ITT_SCOPED_TASK(infer1)
                     _decoder_batch[1]._infer_request.infer();
                  });

               run_first_batch_fut.wait();
               run_second_batch_fut.wait();
            }

            //fill last hidden state tensor
            {
               using namespace torch::indexing;
               _last_hidden_state.index_put_({ 0, Slice(), Slice() }, _decoder_batch[0]._last_hidden_state);
               _last_hidden_state.index_put_({ 1, Slice(), Slice() }, _decoder_batch[1]._last_hidden_state);
            }

            ITT_SCOPED_TASK(update_past_key_values);

            for (int dbi = 0; dbi < 2; dbi++)
            {
               if (_past_length < _decoder_batch[dbi]._past_key_values_torch[0][0].sizes()[2])
               {
                  //insert the new key / value tensors into the past keys tensor
                  for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
                  {
                     for (int i = 0; i < 2; i++)
                     {
                        using namespace torch::indexing;

                        _decoder_batch[dbi]._past_key_values_torch[layeri][i].index_put_({ Slice(), Slice(), Slice(_past_length, _past_length + 1), Slice() }, _decoder_batch[dbi]._new_key_values[layeri][i]);
                     }
                  }
               }
            }

            _past_length++;

            last_hidden_states_ret = _last_hidden_state_ov;
         }

         return last_hidden_states_ret;
      }

      virtual ov::Tensor get_last_hidden_state() override
      {
         return _last_hidden_state_ov;
      }

      virtual int64_t PastLength() override
      {
         return _past_length;
      }

   private:

      struct DecoderBatchStuff
      {
         ov::InferRequest _infer_request;

         //input tensors (OpenVINO tensors & OV Tensors wrapped as torch::Tensor's)
         torch::Tensor _hidden_states;
         ov::Tensor _encoder_attention_mask_ov;
         torch::Tensor _encoder_attention_mask;
         ov::Tensor _custom_attention_mask;

         std::vector< std::vector< ov::Tensor > > _past_key_values_ov;
         std::vector< std::vector< torch::Tensor > > _past_key_values_torch;

         //output tensors
         ov::Tensor _last_hidden_state_ov;
         torch::Tensor _last_hidden_state;  //simply a wrapper around _last_hidden_state_ov (pointing to same underlying buffer)
         std::vector< std::vector< ov::Tensor > > _new_key_values_ov;
         std::vector< std::vector< torch::Tensor > > _new_key_values;
      };

      int64_t _past_length = 0;

      ov::Tensor _last_hidden_state_ov;
      torch::Tensor _last_hidden_state;  //wrapper around _last_hidden_state_ov

      DecoderBatchStuff _decoder_batch[2];

      //model that needs to run when _past_length is 0 to produce initial key/vals
      ov::InferRequest _infer_request_initial;
      torch::Tensor _intiial_encoder_hidden_state;
      std::vector< std::vector< torch::Tensor > > _initial_past_key_values;

      //todo, this should go away:
      torch::Tensor _attention_mask;

      //large context stuff
      ov::InferRequest _infer_request_large_context;
   };
}
