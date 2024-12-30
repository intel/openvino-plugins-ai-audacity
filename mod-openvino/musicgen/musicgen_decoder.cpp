// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#include "musicgen_decoder.h"
#include "musicgen_utils.h"
#include "musicgen_config.h"

#define MAX_PROMPT_TOKENS 64

namespace ov_musicgen
{

   static MusicgenDecoder::Config GenerateDecoderConfig(MusicGenConfig& config)
   {
      MusicgenDecoder::Config decoder_config;
      if (config.bStereo)
      {
         decoder_config.num_codebooks = 8;
      }
      else
      {
         decoder_config.num_codebooks = 4;
      }

      switch (config.model_selection)
      {
      case MusicGenConfig::ModelSelection::MUSICGEN_SMALL_FP16:
      case MusicGenConfig::ModelSelection::MUSICGEN_SMALL_INT8:
         decoder_config.num_hidden_layers = 24;
         decoder_config.num_attention_heads = 16;
         break;

      case MusicGenConfig::ModelSelection::MUSICGEN_MEDIUM_FP16:
      case MusicGenConfig::ModelSelection::MUSICGEN_MEDIUM_INT8:
         decoder_config.num_hidden_layers = 48;
         decoder_config.num_attention_heads = 24;
         break;

      default:
         throw std::runtime_error("Invalid model selection");
         break;
      }

      return decoder_config;
   }

	MusicgenDecoderStatic::MusicgenDecoderStatic(ov::Core& core, MusicGenConfig& config)
      : _decoder_config(GenerateDecoderConfig(config))
	{
		auto model_folder = config.model_folder;

      std::string device = config.musicgen_decode_device0;

      auto tensortype = device == "CPU" ? ov::element::f32 : ov::element::f16;

      //hack for now.
      if (config.model_selection == MusicGenConfig::ModelSelection::MUSICGEN_MEDIUM_FP16 ||
         config.model_selection == MusicGenConfig::ModelSelection::MUSICGEN_MEDIUM_INT8)
      {
         model_folder = FullPath(model_folder, "medium");
      }

      if (config.bStereo)
      {
         model_folder = FullPath(model_folder, "stereo");
      }
      else
      {
         model_folder = FullPath(model_folder, "mono");
      }

      //TODO: This is temporary
      model_folder = FullPath(model_folder, "refactor");
      {
         std::string decoder_model_path;
         switch (config.model_selection)
         {
         case MusicGenConfig::ModelSelection::MUSICGEN_SMALL_FP16:
         case MusicGenConfig::ModelSelection::MUSICGEN_MEDIUM_FP16:
            decoder_model_path = FullPath(model_folder, "musicgen_decoder.xml");
            break;

         case  MusicGenConfig::ModelSelection::MUSICGEN_SMALL_INT8:
         case  MusicGenConfig::ModelSelection::MUSICGEN_MEDIUM_INT8:
            decoder_model_path = FullPath(model_folder, "musicgen_decoder_int8.xml");
            break;

         default:
            throw std::runtime_error("Invalid model selection");
            break;
         }

         std::cout << " Using model=" << decoder_model_path << std::endl;
         std::shared_ptr<ov::Model> model = core.read_model(decoder_model_path);

         //reshape to static shapes
         {
            size_t max_tokens = 1004;
            std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
            port_to_shape[model->input("encoder_attention_mask")] = { 2, MAX_PROMPT_TOKENS };
            port_to_shape[model->input("decoder_input_ids")] = { _decoder_config.num_codebooks*2, 1 };
            port_to_shape[model->input("encoder_hidden_states")] = { 2, MAX_PROMPT_TOKENS, 768 };
            port_to_shape[model->input("decoder_attention_mask")] = { 2, max_tokens + 1 };
            port_to_shape[model->input("past_key_length_tens")] = { 1 };

            for (int enci = 0; enci < _decoder_config.num_hidden_layers; enci++)
            {
               std::string iname = "past_key_values." + std::to_string(enci) + ".encoder.";
               std::string key_name = iname + "key";
               std::string value_name = iname + "value";

               port_to_shape[model->input(key_name)] = { 2, _decoder_config.num_attention_heads, MAX_PROMPT_TOKENS, 64 };
               port_to_shape[model->input(value_name)] = { 2, _decoder_config.num_attention_heads, MAX_PROMPT_TOKENS, 64 };
            }

            for (int deci = 0; deci < _decoder_config.num_hidden_layers; deci++)
            {
               std::string iname = "past_key_values." + std::to_string(deci) + ".decoder.";
               std::string key_name = iname + "key";
               std::string value_name = iname + "value";

               port_to_shape[model->input(key_name)] = { 2, _decoder_config.num_attention_heads, max_tokens, 64 };
               port_to_shape[model->input(value_name)] = { 2, _decoder_config.num_attention_heads, max_tokens, 64 };
            }

            model->reshape(port_to_shape);
         }

         //now, set desired tensor types
         {
            ov::preprocess::PrePostProcessor ppp(model);
            for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
            {
               {
                  std::string past_base_name = "past_key_values." + std::to_string(layeri) + ".decoder.";
                  std::string past_key_name = past_base_name + "key";
                  std::string past_value_name = past_base_name + "value";
                  ppp.input(past_key_name).tensor().set_element_type(tensortype);
                  ppp.input(past_value_name).tensor().set_element_type(tensortype);
               }

               {
                  std::string past_base_name = "past_key_values." + std::to_string(layeri) + ".encoder.";
                  std::string past_key_name = past_base_name + "key";
                  std::string past_value_name = past_base_name + "value";
                  ppp.input(past_key_name).tensor().set_element_type(tensortype);
                  ppp.input(past_value_name).tensor().set_element_type(tensortype);
               }

               {
                  std::string present_base_name = "present." + std::to_string(layeri) + ".decoder.";
                  std::string present_key_name = present_base_name + "key";
                  std::string present_value_name = present_base_name + "value";
                  ppp.output(present_key_name).tensor().set_element_type(tensortype);
                  ppp.output(present_value_name).tensor().set_element_type(tensortype);
               }
            }

            model = ppp.build();
         }

         std::cout << "REFACTORED MODEL:" << std::endl;
         logBasicModelInfo(model);
         ov::serialize(model, "serialized_musicgen_decoder.xml");

         using namespace std::chrono;
         using Clock = std::chrono::high_resolution_clock;

         uint64_t  t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
         ov::CompiledModel compiledModel = core.compile_model(model, device);
         uint64_t  t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
         std::cout << "    compile time = " << (t1 - t0) << " ms" << std::endl;

         _infer_request = compiledModel.create_infer_request();

         //populate kv cache tensor vectors
         for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
         {
            {
               std::string past_base_name = "past_key_values." + std::to_string(layeri) + ".decoder.";
               std::string past_key_name = past_base_name + "key";
               std::string past_value_name = past_base_name + "value";
               past_decoder_keys.push_back(_infer_request.get_tensor(past_key_name));
               past_decoder_values.push_back(_infer_request.get_tensor(past_value_name));
            }

            {
               std::string past_base_name = "past_key_values." + std::to_string(layeri) + ".encoder.";
               std::string past_key_name = past_base_name + "key";
               std::string past_value_name = past_base_name + "value";
               past_encoder_keys.push_back(_infer_request.get_tensor(past_key_name));
               past_encoder_values.push_back(_infer_request.get_tensor(past_value_name));
            }

            {
               std::string present_base_name = "present." + std::to_string(layeri) + ".decoder.";
               std::string present_key_name = present_base_name + "key";
               std::string present_value_name = present_base_name + "value";
               present_decoder_keys.push_back(_infer_request.get_tensor(present_key_name));
               present_decoder_values.push_back(_infer_request.get_tensor(present_value_name));
            }
         }
      }

      //prep cross attn kv producer model. This is the thing that generates the
      // encoder KV's
      {
         auto model_path = FullPath(model_folder, "initial_cross_attn_kv_producer.xml");
         std::shared_ptr<ov::Model> model = core.read_model(model_path);

         std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
         port_to_shape[model->input("encoder_hidden_states")] = { 2, past_encoder_keys[0].get_shape()[2], 768};
         port_to_shape[model->input("attention_mask")] = { 2, past_encoder_keys[0].get_shape()[2] };
         model->reshape(port_to_shape);

         ov::preprocess::PrePostProcessor ppp(model);
         for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
         {
            std::string present_base_name = "present." + std::to_string(layeri) + ".encoder.";
            std::string present_key_name = present_base_name + "key";
            std::string present_value_name = present_base_name + "value";

            ppp.output(present_key_name).tensor().set_element_type(tensortype);
            ppp.output(present_value_name).tensor().set_element_type(tensortype);
         }

         model = ppp.build();

         std::cout << "REFACTORED INITAL CROSS ATTN KV PRODUCER MODEL:" << std::endl;
         logBasicModelInfo(model);

         ov::CompiledModel compiledModel = core.compile_model(model, device);

         _infer_request_initial = compiledModel.create_infer_request();

         // share some tensors between initial kv model, and decoder model
         _infer_request_initial.set_tensor("attention_mask", _infer_request.get_tensor("encoder_attention_mask"));

         for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
         {
            std::string present_base_name = "present." + std::to_string(layeri) + ".encoder.";
            std::string present_key_name = present_base_name + "key";
            std::string present_value_name = present_base_name + "value";

            _infer_request_initial.set_tensor(present_key_name, past_encoder_keys[layeri]);
            _infer_request_initial.set_tensor(present_value_name, past_encoder_values[layeri]);
         }

         _infer_request_initial.infer();
      }


      //prep large context model
      {
         std::string decoder_model_path;
         switch (config.model_selection)
         {
         case MusicGenConfig::ModelSelection::MUSICGEN_SMALL_FP16:
         case MusicGenConfig::ModelSelection::MUSICGEN_MEDIUM_FP16:
            decoder_model_path = FullPath(model_folder, "musicgen_decoder_nonkv.xml");
            break;

         case  MusicGenConfig::ModelSelection::MUSICGEN_SMALL_INT8:
         case  MusicGenConfig::ModelSelection::MUSICGEN_MEDIUM_INT8:
            decoder_model_path = FullPath(model_folder, "musicgen_decoder_nonkv_int8.xml");
            break;

         default:
            throw std::runtime_error("Invalid model selection");
            break;
         }

         std::shared_ptr<ov::Model> model = core.read_model(decoder_model_path);

         size_t num_ids;
         switch (config.m_continuation_context)
         {
            case MusicGenConfig::ContinuationContext::FIVE_SECONDS:
            {
               num_ids = 251; //5 seconds * 50 hz + 1
            }
            break;

            case MusicGenConfig::ContinuationContext::TEN_SECONDS:
            {
               num_ids = 501; //10 seconds * 50 hz + 1
            }
            break;

            default:
               throw std::runtime_error("invalid MusicGenConfig::ContinuationContext");
            break;
         }

         //reshape to static shapes
         {
            std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
            port_to_shape[model->input("encoder_attention_mask")] = { 2, MAX_PROMPT_TOKENS };
            port_to_shape[model->input("decoder_input_ids")] = { _decoder_config.num_codebooks * 2, num_ids };
            port_to_shape[model->input("encoder_hidden_states")] = { 2, MAX_PROMPT_TOKENS, 768 };

            for (int enci = 0; enci < _decoder_config.num_hidden_layers; enci++)
            {
               std::string iname = "past_key_values." + std::to_string(enci) + ".encoder.";
               std::string key_name = iname + "key";
               std::string value_name = iname + "value";

               port_to_shape[model->input(key_name)] = { 2, _decoder_config.num_attention_heads, MAX_PROMPT_TOKENS, 64 };
               port_to_shape[model->input(value_name)] = { 2, _decoder_config.num_attention_heads, MAX_PROMPT_TOKENS, 64 };
            }

            model->reshape(port_to_shape);
         }

         //now, set desired tensor types
         {
            ov::preprocess::PrePostProcessor ppp(model);
            for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
            {
               {
                  std::string past_base_name = "past_key_values." + std::to_string(layeri) + ".encoder.";
                  std::string past_key_name = past_base_name + "key";
                  std::string past_value_name = past_base_name + "value";
                  ppp.input(past_key_name).tensor().set_element_type(tensortype);
                  ppp.input(past_value_name).tensor().set_element_type(tensortype);
               }

               {
                  std::string present_base_name = "present." + std::to_string(layeri) + ".decoder.";
                  std::string present_key_name = present_base_name + "key";
                  std::string present_value_name = present_base_name + "value";
                  ppp.output(present_key_name).tensor().set_element_type(tensortype);
                  ppp.output(present_value_name).tensor().set_element_type(tensortype);
               }
            }

            model = ppp.build();
         }

         std::cout << "REFACTORED NON-KV MODEL:" << std::endl;
         logBasicModelInfo(model);
         ov::serialize(model, "serialized_musicgen_decoder_nonkv.xml");

         ov::CompiledModel compiledModel = core.compile_model(model, device);

         _infer_request_nonkv = compiledModel.create_infer_request();

         for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
         {
            std::string past_base_name = "past_key_values." + std::to_string(layeri) + ".encoder.";
            std::string past_key_name = past_base_name + "key";
            std::string past_value_name = past_base_name + "value";

            _infer_request_nonkv.set_tensor(past_key_name, past_encoder_keys[layeri]);
            _infer_request_nonkv.set_tensor(past_value_name, past_encoder_values[layeri]);
         }

         for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
         {

            {
               std::string present_base_name = "present." + std::to_string(layeri) + ".decoder.";
               std::string present_key_name = present_base_name + "key";
               std::string present_value_name = present_base_name + "value";
               present_decoder_keys_large_context.push_back(_infer_request_nonkv.get_tensor(present_key_name));
               present_decoder_values_large_context.push_back(_infer_request_nonkv.get_tensor(present_value_name));
            }
         }

         _infer_request_nonkv.infer();
      }

      Reset();
      std::cout << "MusicgenDecoderStatic construction complete!" << std::endl;
	}

   static inline void clear_tens_vec(std::vector< ov::Tensor > &t_vec)
   {
      for (auto &tensor : t_vec)
      {
         void* pData = tensor.data();
          std::memset(pData, 0, tensor.get_byte_size());
      }
   }

   void MusicgenDecoderStatic::Reset()
   {
      _past_length = 0;

      //reset kv cache tensors to 0
      clear_tens_vec(past_decoder_keys);
      clear_tens_vec(past_decoder_values);


      clear_tens_vec(present_decoder_keys);
      clear_tens_vec(present_decoder_values);

      clear_tens_vec(present_decoder_keys_large_context);
      clear_tens_vec(present_decoder_values_large_context);

      //clear initial input tensor
      {
         auto encoder_hidden_states = _infer_request_initial.get_tensor("encoder_hidden_states");
         memset(encoder_hidden_states.data(), 0, encoder_hidden_states.get_byte_size());
      }

      {
         auto encoder_attention_mask = _infer_request_initial.get_tensor("attention_mask");
         memset(encoder_attention_mask.data(), 0, encoder_attention_mask.get_byte_size());
      }

      //clear large context tensors
      {
         auto encoder_hidden_states = _infer_request_nonkv.get_tensor("encoder_hidden_states");
         memset(encoder_hidden_states.data(), 0, encoder_hidden_states.get_byte_size());
      }

      {
         auto encoder_attention_mask = _infer_request_nonkv.get_tensor("encoder_attention_mask");
         memset(encoder_attention_mask.data(), 0, encoder_attention_mask.get_byte_size());
      }
      
      //fill attention masks with 0's
      for (auto& tensorname : { "encoder_attention_mask", "decoder_attention_mask" })
      {
         auto tensor = _infer_request.get_tensor(tensorname);
         memset(tensor.data(), 0, tensor.get_byte_size());
      }
   }

   torch::Tensor MusicgenDecoderStatic::run(torch::Tensor input_ids,
		std::optional<torch::Tensor> encoder_hidden_states,
		std::optional<torch::Tensor> encoder_attention_mask)
	{
      using namespace torch::indexing;
      if (_past_length == 0 && encoder_hidden_states && encoder_attention_mask)
      {
         auto input_encoder_attention_mask = wrap_ov_tensor_as_torch(_infer_request_initial.get_tensor("attention_mask"));
         auto encoder_hidden_states_ov = wrap_ov_tensor_as_torch(_infer_request_initial.get_tensor("encoder_hidden_states"));

         input_encoder_attention_mask.index_put_({ Slice(), Slice(0, encoder_attention_mask->sizes()[1]) }, *encoder_attention_mask);
         encoder_hidden_states_ov.index_put_({ Slice(), Slice(0, encoder_hidden_states->sizes()[1]), Slice() }, *encoder_hidden_states);

         //run inference, producing encoder kv values.
         _infer_request_initial.infer();
      }

      torch::Tensor logits;

      if (input_ids.sizes()[1] > 1)
      {
         //set input ids
         auto decoder_input_ids = wrap_ov_tensor_as_torch(_infer_request_nonkv.get_tensor("decoder_input_ids"));
         decoder_input_ids.copy_(input_ids);

         //set encoder attention mask
         auto encoder_attention_mask_ov = wrap_ov_tensor_as_torch(_infer_request_nonkv.get_tensor("encoder_attention_mask"));
         encoder_attention_mask_ov.index_put_({ Slice(), Slice(0, encoder_attention_mask->sizes()[1]) }, *encoder_attention_mask);

         //set encoder hidden states
         auto encoder_hidden_states_ov = wrap_ov_tensor_as_torch(_infer_request_nonkv.get_tensor("encoder_hidden_states"));
         encoder_hidden_states_ov.index_put_({ Slice(), Slice(0, encoder_hidden_states->sizes()[1]), Slice() }, *encoder_hidden_states);

         _infer_request_nonkv.infer();

         //copy 'kv_valid_size' kv values  
         for (int deci = 0; deci < _decoder_config.num_hidden_layers; deci++)
         {
            auto past_key_tensor = past_decoder_keys[deci];
            auto past_value_tensor = past_decoder_values[deci];

            auto present_key_tensor = present_decoder_keys_large_context[deci];
            auto present_value_tensor = present_decoder_values_large_context[deci];

            size_t kv_valid_size = present_key_tensor.get_shape()[2];

            const ov::Coordinate begin = { 0, 0, 0, 0 };
            const ov::Coordinate end = { 2, _decoder_config.num_attention_heads, kv_valid_size, 64 };
            auto past_key_slice = ov::Tensor(past_key_tensor, begin, end);
            auto past_value_slice = ov::Tensor(past_value_tensor, begin, end);

            present_key_tensor.copy_to(past_key_slice);
            present_value_tensor.copy_to(past_value_slice);
         }

         logits = wrap_ov_tensor_as_torch(_infer_request_nonkv.get_tensor("logits"));

         _past_length = logits.sizes()[1];

         auto decoder_attention_mask = wrap_ov_tensor_as_torch(_infer_request.get_tensor("decoder_attention_mask"));
         decoder_attention_mask.index_put_({ torch::indexing::Slice(),  torch::indexing::Slice(0,_past_length) }, 1);
      }
      else
      {
         //set input ids
         auto decoder_input_ids = wrap_ov_tensor_as_torch(_infer_request.get_tensor("decoder_input_ids"));
         decoder_input_ids.copy_(input_ids);

         //set encoder attention mask
         auto encoder_attention_mask_ov = wrap_ov_tensor_as_torch(_infer_request.get_tensor("encoder_attention_mask"));
         encoder_attention_mask_ov.index_put_({ Slice(), Slice(0, encoder_attention_mask->sizes()[1]) }, *encoder_attention_mask);

         //set encoder hidden states
         auto encoder_hidden_states_ov = wrap_ov_tensor_as_torch(_infer_request.get_tensor("encoder_hidden_states"));
         encoder_hidden_states_ov.index_put_({ Slice(), Slice(0, encoder_hidden_states->sizes()[1]), Slice() }, *encoder_hidden_states);

         //set decoder attention mask
         {

            auto decoder_attention_mask = wrap_ov_tensor_as_torch(_infer_request.get_tensor("decoder_attention_mask"));
            
            if (_past_length > 0)
            {
               decoder_attention_mask.index_put_({ torch::indexing::Slice(), _past_length - 1 }, 1);
            }

            decoder_attention_mask.index_put_({ torch::indexing::Slice(), -1 }, 1);
         }

         //set past_key_length_tens
         {
            auto past_key_length_tens = _infer_request.get_tensor("past_key_length_tens");
            int64_t* pPastKeyLength = past_key_length_tens.data<int64_t>();
            *pPastKeyLength = _past_length;
         }

         _infer_request.infer();

         for (int deci = 0; deci < _decoder_config.num_hidden_layers; deci++)
         {
            std::string iname = "past_key_values." + std::to_string(deci) + ".decoder.";
            std::string key_name = iname + "key";
            std::string value_name = iname + "value";

            auto past_key_tensor = _infer_request.get_tensor(key_name);
            auto past_value_tensor = _infer_request.get_tensor(value_name);

            const ov::Coordinate begin = { 0, 0, (size_t)_past_length, 0 };
            const ov::Coordinate end = { 2, _decoder_config.num_attention_heads, (size_t)_past_length + 1, 64 };
            auto past_key_slice = ov::Tensor(past_key_tensor, begin, end);
            auto past_value_slice = ov::Tensor(past_value_tensor, begin, end);

            iname = "present." + std::to_string(deci) + ".decoder.";
            key_name = iname + "key";
            value_name = iname + "value";

            auto present_key_tensor = _infer_request.get_tensor(key_name);
            auto present_value_tensor = _infer_request.get_tensor(value_name);

            present_key_tensor.copy_to(past_key_slice);
            present_value_tensor.copy_to(past_value_slice);
         }

         logits = wrap_ov_tensor_as_torch(_infer_request.get_tensor("logits"));

         _past_length++;
      }

		return logits;
	}

	int64_t MusicgenDecoderStatic::PastLength()
	{
		return _past_length;
	}

	int64_t MusicgenDecoderStatic::MaxNewTokens()
	{
		return past_decoder_keys[0].get_shape()[2];
	}
}
