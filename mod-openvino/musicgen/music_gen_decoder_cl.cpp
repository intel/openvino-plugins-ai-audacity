// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#include "music_gen_decoder_cl.h"
#include <ittutils.h>
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include <future>

namespace ov_musicgen
{
   struct MusicgenDecoderModelCL::CLStuff
   {
      cl::Context context;
      cl::CommandQueue queue;
      std::vector< std::vector< cl::Buffer > > _past_key_values_cl;
      std::vector< std::vector< cl::Buffer > > _new_key_values_cl;
      std::vector< std::vector< cl::Buffer > > _new_key_values_large_cl;
      std::future<void> update_past_keys_fut;
   };

   MusicgenDecoderModelCL::MusicgenDecoderModelCL(ov::Core& core, MusicGenConfig& config)
   {
      auto model_folder = config.model_folder;

      _cl_stuff = std::make_shared< CLStuff >();

      std::string device = config.musicgen_decode_device0;

      auto gpu_context = core.get_default_context(device).as<ov::intel_gpu::ocl::ClContext>();

      auto tensortype = device == "CPU" ? ov::element::f32 : ov::element::f16;

      if (config.bStereo)
      {
         model_folder = FullPath(model_folder, "stereo");
      }
      else
      {
         model_folder = FullPath(model_folder, "mono");
      }

      //prep decoder model
      {
         std::string decoder_model_path, binfile;
         switch (config.model_selection)
         {
         case MusicGenConfig::ModelSelection::MUSICGEN_SMALL_FP16:
            decoder_model_path = FullPath(model_folder, "musicgen_decoder_static.xml");
            binfile = FullPath(model_folder, "musicgen_decoder_combined_weights.bin");
            break;

         case  MusicGenConfig::ModelSelection::MUSICGEN_SMALL_INT8:
            decoder_model_path = FullPath(model_folder, "musicgen_decoder_static_int8.xml");
            binfile = FullPath(model_folder, "musicgen_decoder_combined_weights_int8.bin");
            break;

         default:
            throw std::runtime_error("Invalid model selection");
            break;
         }

         std::cout << " Using model=" << decoder_model_path << ", " << binfile << std::endl;

         std::shared_ptr<ov::Model> model = core.read_model(decoder_model_path, binfile);

         {
            size_t max_tokens = 1004;
            std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
            port_to_shape[model->input("input_hidden_states")] = { 2, 1, 1024 };
            port_to_shape[model->input("encoder_attention_mask")] = { 2, 1, 1, 64 };
            port_to_shape[model->input("custom_attention_mask")] = { 1, max_tokens + 1 };

            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (size_t i = 0; i < 4; i++)
               {
                  std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

                  if (i < 2)
                  {
                     port_to_shape[model->input(tensorname)] = { 2, 16, max_tokens, 64 };
                  }
                  else
                  {
                     port_to_shape[model->input(tensorname)] = { 2, 16, 64, 64 };
                  }
               }
            }

            model->reshape(port_to_shape);
         }

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

         logBasicModelInfo(model);

         using namespace std::chrono;
         using Clock = std::chrono::high_resolution_clock;

         uint64_t  t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
         ov::CompiledModel compiledModel;
         if (config.performance_hint)
         {
            auto mode = get_performance_hint(device, *config.performance_hint, core);
            compiledModel = core.compile_model(model, device, ov::hint::performance_mode(mode));
         }
         else
         {
            compiledModel = core.compile_model(model, device);
         }
         uint64_t  t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
         std::cout << "    compile time = " << (t1 - t0) << " ms" << std::endl;

         std::cout << "static decoder model properties:" << std::endl;
         print_compiled_model_properties(compiledModel);

         _infer_request = compiledModel.create_infer_request();

         _cl_stuff->context = gpu_context.get();
         cl::Device cl_device = cl::Device(_cl_stuff->context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);
         cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
         _cl_stuff->queue = cl::CommandQueue(_cl_stuff->context, cl_device, props);

         //allocate an OpenCL buffer for each of the input past_key_value tensors
         _cl_stuff->_past_key_values_cl.resize(N_LAYERS);
         _cl_stuff->_new_key_values_cl.resize(N_LAYERS);
         for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
         {
            _cl_stuff->_past_key_values_cl[layeri].resize(4);

            for (size_t i = 0; i < 4; i++)
            {
               std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
               auto ov_tensor = _infer_request.get_tensor(tensorname);
               auto tensor_byte_size = ov_tensor.get_byte_size();
               cl_int err;
               cl::Buffer cl_buf(_cl_stuff->context, CL_MEM_READ_WRITE, tensor_byte_size, NULL, &err);
               auto shared_blob = gpu_context.create_tensor(ov_tensor.get_element_type(), ov_tensor.get_shape(), cl_buf);
               //replace the default tensor.
               _infer_request.set_tensor(tensorname, shared_blob);
               _cl_stuff->_past_key_values_cl[layeri][i] = cl_buf;
            }

            _cl_stuff->_new_key_values_cl[layeri].resize(2);

            for (int i = 0; i < 2; i++)
            {
               std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
               auto ov_tensor = _infer_request.get_tensor(tensorname);
               auto tensor_byte_size = ov_tensor.get_byte_size();
               cl_int err;
               cl::Buffer cl_buf(_cl_stuff->context, CL_MEM_READ_WRITE, tensor_byte_size, NULL, &err);
               auto shared_blob = gpu_context.create_tensor(ov_tensor.get_element_type(), ov_tensor.get_shape(), cl_buf);
               //replace the default tensor.
               _infer_request.set_tensor(tensorname, shared_blob);
               _cl_stuff->_new_key_values_cl[layeri][i] = cl_buf;

            }
         }

         //first inference is usually a bit longer due to some lazy initialization, so trigger a warm up inference.
         _infer_request.infer();

         _hidden_states = wrap_ov_tensor_as_torch(_infer_request.get_tensor("input_hidden_states"));
         _encoder_attention_mask_ov = _infer_request.get_tensor("encoder_attention_mask");
         _encoder_attention_mask = wrap_ov_tensor_as_torch(_encoder_attention_mask_ov);

         _custom_attention_mask = _infer_request.get_tensor("custom_attention_mask");

         _past_key_values_ov.resize(N_LAYERS);
         for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
         {
            _past_key_values_ov[layeri].resize(4);

            for (size_t i = 0; i < 4; i++)
            {
               std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

               _past_key_values_ov[layeri][i] = _infer_request.get_tensor(tensorname);
            }
         }

         //get last hidden state tensor
         _last_hidden_state_ov = _infer_request.get_tensor("last_hidden_state");

         //wrap it as a torch::Tensor
         _last_hidden_state = wrap_ov_tensor_as_torch(_last_hidden_state_ov);

         _new_key_values_ov.resize(N_LAYERS);
         for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
         {
            _new_key_values_ov[layeri].resize(2);

            for (int i = 0; i < 2; i++)
            {
               std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);


               auto ov_tensor = _infer_request.get_tensor(tensorname);

               _new_key_values_ov[layeri][i] = ov_tensor;
            }
         }

         std::cout << "Batch2 Decoder" << std::endl;
         std::cout << "    tensortype = " << tensortype << std::endl;
         std::cout << "    max token length = " << MaxNewTokens() << std::endl;
      }

      //prep initial model
      {
         auto model_path = FullPath(model_folder, "initial_cross_attn_kv_producer.xml");
         std::shared_ptr<ov::Model> model = core.read_model(model_path);

         model->reshape({ {2, _past_key_values_ov[0][2].get_shape()[2], 1024} });

         ov::preprocess::PrePostProcessor ppp(model);

         for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
         {
            for (int i = 0; i < 2; i++)
            {
               std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
               ppp.output(tensorname).tensor().set_element_type(tensortype);

            }
         }

         model = ppp.build();

         ov::CompiledModel compiledModel = core.compile_model(model, device);

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

               //link the initial model with the decode model
               _infer_request_initial.set_tensor(tensorname, _past_key_values_ov[layeri][i + 2]);
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
               ppp.input(tensorname).tensor().set_element_type(tensortype);
            }

            for (int i = 0; i < 2; i++)
            {
               std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
               ppp.output(tensorname).tensor().set_element_type(tensortype);
            }
         }

         model = ppp.build();

         std::cout << "large context model" << std::endl;
         logBasicModelInfo(model);

         ov::CompiledModel compiledModel = core.compile_model(model, device);

         _infer_request_large_context = compiledModel.create_infer_request();
         _infer_request_large_context.infer();

         _cl_stuff->_new_key_values_large_cl.resize(N_LAYERS);
         for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
         {
            _cl_stuff->_new_key_values_large_cl[layeri].resize(2);
            for (int i = 0; i < 2; i++)
            {
               {
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                  auto ov_tensor = _infer_request_large_context.get_tensor(tensorname);
                  auto tensor_byte_size = ov_tensor.get_byte_size();
                  cl_int err;
                  cl::Buffer cl_buf(_cl_stuff->context, CL_MEM_READ_WRITE, tensor_byte_size, NULL, &err);
                  auto shared_blob = gpu_context.create_tensor(ov_tensor.get_element_type(), ov_tensor.get_shape(), cl_buf);
                  //replace the default tensor.
                  _infer_request_large_context.set_tensor(tensorname, shared_blob);
                  _cl_stuff->_new_key_values_large_cl[layeri][i] = cl_buf;
               }

               {
                  std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i + 2);

                  //link the initial model with the this model
                  _infer_request_large_context.set_tensor(tensorname, _past_key_values_ov[layeri][i + 2]);
               }
            }
         }
      }

      Reset();
   }

   int64_t MusicgenDecoderModelCL::MaxNewTokens()
   {
      return _past_key_values_ov[0][0].get_shape()[2];
   }

   void MusicgenDecoderModelCL::Reset()
   {
      _past_length = 0;

      for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
      {
         for (size_t i = 0; i < 4; i++)
         {
            int pattern = 0;
            auto buf = _cl_stuff->_past_key_values_cl[layeri][i];
            size_t size = 0;
            buf.getInfo(CL_MEM_SIZE, &size);
            _cl_stuff->queue.enqueueFillBuffer(buf, pattern, 0, size);
         }
      }
      _cl_stuff->queue.finish();

      memset(_infer_request_initial.get_input_tensor().data<float>(), 0, _infer_request_initial.get_input_tensor().get_byte_size());

      {
         float* pAttnMask = _custom_attention_mask.data<float>();
         //std::cout << "_custom_attention_mask.get_size() = " << _custom_attention_mask.get_size() << std::endl;
         for (int i = 0; i < _custom_attention_mask.get_size(); i++)
         {
            pAttnMask[i] = -INFINITY;
         }
      }

      {
         float* pAttnMask = _encoder_attention_mask_ov.data<float>();
         for (int i = 0; i < _encoder_attention_mask_ov.get_size(); i++)
         {
            pAttnMask[i] = -INFINITY;
         }
      }
   }

   void MusicgenDecoderModelCL::ShiftLeft(int64_t ntokens)
   {

   }

   ov::Tensor MusicgenDecoderModelCL::run(torch::Tensor hidden_states, std::optional<torch::Tensor> encoder_hidden_states, std::optional<torch::Tensor> encoder_attention_mask)
   {
      ITT_SCOPED_TASK(MusicgenModelStatic_run)
         ov::Tensor last_hidden_states_ret;
      if (_past_length == 0)
      {
         ITT_SCOPED_TASK(initial_infer)
            using namespace torch::indexing;
         _intiial_encoder_hidden_state.index_put_({ Slice(), Slice(0, encoder_hidden_states->sizes()[1]), Slice() }, *encoder_hidden_states);
         _infer_request_initial.infer();
      }

      if (hidden_states.sizes()[1] > 1)
      {
         ITT_SCOPED_TASK(large_context_path)
            using namespace torch::indexing;


         auto input_hidden_states = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor("input_hidden_states"));
         auto input_attention_mask = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor("attention_mask"));
         auto input_encoder_attention_mask = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor("encoder_attention_mask"));

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

         ITT_SCOPED_TASK(update_past_key_values);

         int64_t valid_height = ov_last_hidden_state.get_shape()[1];

         //insert the new key / value tensors into the past keys tensor
         auto past_key_values_shape = _past_key_values_ov[0][0].get_shape();
         auto new_key_values_shape = _infer_request_large_context.get_tensor("new_key_value_0_0").get_shape();
         for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
         {
            for (int i = 0; i < 2; i++)
            {
               //slice the new key values into the past_key_vals buffer using OpenCL.
               std::array<size_t, 3> srcOrigin = { 0, 0, 0 }; // Start at the beginning of the source buffer
               std::array<size_t, 3> dstOrigin = { 0, 0, 0 }; // Start at the beginning of the destination buffer

               // Size of one element
               std::array<size_t, 3> region = { sizeof(ov::float16) * past_key_values_shape[3], new_key_values_shape[2], past_key_values_shape[0] * past_key_values_shape[1] };

               size_t srcRowPitch = past_key_values_shape[3] * sizeof(ov::float16); // Size of one row in the source buffer
               size_t srcSlicePitch = srcRowPitch * new_key_values_shape[2]; // Size of one 2D plane in the source buffer
               size_t dstRowPitch = past_key_values_shape[3] * sizeof(ov::float16); // Size of one row in the destination buffer
               size_t dstSlicePitch = dstRowPitch * past_key_values_shape[2]; // Size of one 2D plane in the destination buffer

               auto new_key_values = _cl_stuff->_new_key_values_large_cl[layeri][i];
               auto past_key_values = _cl_stuff->_past_key_values_cl[layeri][i];

               cl_int ret = _cl_stuff->queue.enqueueCopyBufferRect(new_key_values, past_key_values, srcOrigin, dstOrigin, region, srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch);
            }
         }

         _cl_stuff->queue.finish();

         float* pAttnMask = _custom_attention_mask.data<float>();
         memset(pAttnMask, 0, valid_height * sizeof(float));
         _past_length = valid_height;
      }
      else
      {
         //set attention mask
         float* pAttnMask = _custom_attention_mask.data<float>();
         if (_past_length > 0)
         {
            pAttnMask[_past_length - 1] = 0.f;
         }

         pAttnMask[_custom_attention_mask.get_size() - 1] = 0.f;

         //set input tensors
         {
            using namespace torch::indexing;
            _hidden_states.index_put_({ Slice(), Slice(), Slice() }, hidden_states);

            if (encoder_attention_mask)
            {
               _encoder_attention_mask.index_put_({ Slice(), Slice(), Slice(), Slice(0, encoder_attention_mask->sizes()[3]) }, *encoder_attention_mask);
            }
         }

         if (_cl_stuff->update_past_keys_fut.valid())
         {
            _cl_stuff->update_past_keys_fut.wait();
         }

         {
            ITT_SCOPED_TASK(infer)
               _infer_request.infer();
         }

         last_hidden_states_ret = _last_hidden_state_ov;

         if (_past_length < _past_key_values_ov[0][0].get_shape()[2])
         {
            ITT_SCOPED_TASK(update_past_key_values)

               auto past_key_values_shape = _past_key_values_ov[0][0].get_shape();
            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (int i = 0; i < 2; i++)
               {
                  //slice the new key values into the existing past_key_vals buffer using OpenCL.
                  std::array<size_t, 3> srcOrigin = { 0, 0, 0 }; // Start at the beginning of the source buffer
                  std::array<size_t, 3> dstOrigin = { 0, _past_length,  0 };

                  // Size of one element
                  std::array<size_t, 3> region = { sizeof(ov::float16) * past_key_values_shape[3], 1, past_key_values_shape[0] * past_key_values_shape[1] };

                  size_t srcRowPitch = 64 * sizeof(ov::float16); // Size of one row in the source buffer
                  size_t srcSlicePitch = 64 * sizeof(ov::float16) * 1; // Size of one 2D plane in the source buffer
                  size_t dstRowPitch = 64 * sizeof(ov::float16); // Size of one row in the destination buffer
                  size_t dstSlicePitch = 64 * sizeof(ov::float16) * past_key_values_shape[2]; // Size of one 2D plane in the destination buffer

                  auto new_key_values = _cl_stuff->_new_key_values_cl[layeri][i];
                  auto past_key_values = _cl_stuff->_past_key_values_cl[layeri][i];

                  cl_int ret = _cl_stuff->queue.enqueueCopyBufferRect(new_key_values, past_key_values, srcOrigin, dstOrigin, region, srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch);

               }
            }

            _cl_stuff->queue.finish();

            _past_length++;
         }
         else
         {
            throw std::runtime_error("past key length exceeded it's max length");
         }
      }

      return last_hidden_states_ret;
   }

   ov::Tensor MusicgenDecoderModelCL::get_last_hidden_state()
   {
      return _last_hidden_state_ov;
   }

   int64_t MusicgenDecoderModelCL::PastLength()
   {
      return _past_length;
   }
}
