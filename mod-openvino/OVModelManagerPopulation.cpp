#include "OVModelManager.h"

static std::shared_ptr< OVModelManager::ModelCollection > populate_music_separation()
{
   std::shared_ptr<OVModelManager::ModelInfo> demucs_model_info = std::make_shared<OVModelManager::ModelInfo>();
   demucs_model_info->model_name = "Demucs v4";
   demucs_model_info->info = "Demucs-v4 is a state-of-the-art music source separation model that can separate drums, bass, vocals, and other stems from any song.";
   demucs_model_info->baseUrl = "https://huggingface.co/Intel/demucs-openvino/resolve/97fc578fb57650045d40b00bc84c7d156be77547/";
   demucs_model_info->fileList = { "htdemucs_v4.bin", "htdemucs_v4.xml" };

   auto music_sep_collection = std::make_shared< OVModelManager::ModelCollection >();
   music_sep_collection->models.emplace_back(demucs_model_info);

   return music_sep_collection;
}

static std::shared_ptr< OVModelManager::ModelCollection > populate_music_generation()
{
   //TODO: Change 'main' to specific commit-id
   std::string baseUrl = "https://huggingface.co/Intel/musicgen-static-openvino/resolve/main/";
   std::shared_ptr<OVModelManager::ModelInfo> common = std::make_shared<OVModelManager::ModelInfo>();
   common->model_name = "Music Generation Common";
   common->baseUrl = baseUrl;
   common->relative_path = "musicgen";
   common->fileList = { "musicgen-small-tokenizer.bin", "musicgen-small-tokenizer.xml",
                        "t5.bin", "t5.xml",
                        "openvino_encodec_decode.xml", "openvino_encodec_decode.bin",
                        "openvino_encodec_encode.xml", "openvino_encodec_encode.bin" };

   auto collection = std::make_shared< OVModelManager::ModelCollection >();

   std::vector<std::string> f16_file_list = { "musicgen_decoder.xml", "musicgen_decoder_nonkv.xml", "musicgen_decoder_combined.bin"};
   std::vector<std::string> int8_file_list = { "musicgen_decoder_int8.xml", "musicgen_decoder_nonkv_int8.xml", "musicgen_decoder_int8_combined.bin" };
   std::vector<std::string> cross_attn_common_file_list = { "initial_cross_attn_kv_producer.bin", "initial_cross_attn_kv_producer.xml" };

   //small-mono
   {
      std::shared_ptr<OVModelManager::ModelInfo> cross_attn_common = std::make_shared<OVModelManager::ModelInfo>();
      cross_attn_common->model_name = "Music Generation Cross-Attn Common";
      cross_attn_common->baseUrl = baseUrl;
      cross_attn_common->relative_path = "musicgen";
      cross_attn_common->fileList = cross_attn_common_file_list;
      for (auto& f : cross_attn_common->fileList) {
         f = "small-mono/" + f;
      }

      //small-mono F16
      {
         std::shared_ptr<OVModelManager::ModelInfo> model = std::make_shared<OVModelManager::ModelInfo>();
         model->model_name = "Small Mono (FP16)";
         model->info = "FP16-quantized variant of facebook/musicgen-small model. This is a mono model, therefore it will produce a mono track.";
         model->baseUrl = baseUrl;
         model->relative_path = "musicgen";
         model->dependencies.push_back(common);
         model->dependencies.push_back(cross_attn_common);
         model->fileList = f16_file_list;
         for (auto& f : model->fileList) {
            f = "small-mono/" + f;
         }
         collection->models.emplace_back(model);
      }

      //small-mono INT8
      {
         std::shared_ptr<OVModelManager::ModelInfo> model = std::make_shared<OVModelManager::ModelInfo>();
         model->model_name = "Small Mono (INT8)";
         model->info = "INT8-quantized variant of facebook/musicgen-small model. This is a mono model, therefore it will produce a mono track.";
         model->baseUrl = baseUrl;
         model->relative_path = "musicgen";
         model->dependencies.push_back(common);
         model->dependencies.push_back(cross_attn_common);
         model->fileList = int8_file_list;
         for (auto& f : model->fileList) {
            f = "small-mono/" + f;
         }
         collection->models.emplace_back(model);
      }
   }

   //small-stereo
   {
      std::shared_ptr<OVModelManager::ModelInfo> cross_attn_common = std::make_shared<OVModelManager::ModelInfo>();
      cross_attn_common->model_name = "Music Generation Cross-Attn Common";
      cross_attn_common->baseUrl = baseUrl;
      cross_attn_common->relative_path = "musicgen";
      cross_attn_common->fileList = cross_attn_common_file_list;
      for (auto& f : cross_attn_common->fileList) {
         f = "small-stereo/" + f;
      }

      //small-stereo F16
      {
         std::shared_ptr<OVModelManager::ModelInfo> model = std::make_shared<OVModelManager::ModelInfo>();
         model->model_name = "Small Stereo (FP16)";
         model->info = "FP16-quantized variant of facebook/musicgen-stereo-small model. This is a stereo model, therefore it will produce a stereo track.";
         model->baseUrl = baseUrl;
         model->relative_path = "musicgen";
         model->dependencies.push_back(common);
         model->dependencies.push_back(cross_attn_common);
         model->fileList = f16_file_list;
         for (auto& f : model->fileList) {
            f = "small-stereo/" + f;
         }
         collection->models.emplace_back(model);
      }

      //small-stereo INT8
      {
         std::shared_ptr<OVModelManager::ModelInfo> model = std::make_shared<OVModelManager::ModelInfo>();
         model->model_name = "Small Stereo (INT8)";
         model->info = "INT8-quantized variant of facebook/musicgen-stereo-small model. This is a stereo model, therefore it will produce a stereo track.";
         model->baseUrl = baseUrl;
         model->relative_path = "musicgen";
         model->dependencies.push_back(common);
         model->dependencies.push_back(cross_attn_common);
         model->fileList = int8_file_list;
         for (auto& f : model->fileList) {
            f = "small-stereo/" + f;
         }
         collection->models.emplace_back(model);
      }
   }

   return collection;
}

struct WhisperInfo
{
   std::string ui_name;
   std::string relative_path;
   std::string base_url;
};

static std::shared_ptr< OVModelManager::ModelCollection > populate_whisper()
{
   const std::vector< WhisperInfo> whisper_model_info
   {
      {
         "Whisper Base (FP16)",
         "whisper-base-fp16-ov",
         "https://huggingface.co/OpenVINO/whisper-base-fp16-ov/resolve/main/"
      },
      {
         "Whisper Base (INT8)",
         "whisper-base-int8-ov",
         "https://huggingface.co/OpenVINO/whisper-base-int8-ov/resolve/main/"
      },
      {
         "Whisper Base (INT4)",
         "whisper-base-int4-ov",
         "https://huggingface.co/OpenVINO/whisper-base-int4-ov/resolve/main/"
      },
      {
         "Whisper Medium (FP16)",
         "whisper-medium-fp16-ov",
         "https://huggingface.co/OpenVINO/whisper-medium-fp16-ov/resolve/main/"
      },
      {
         "Whisper Medium (INT8)",
         "whisper-medium-int8-ov",
         "https://huggingface.co/OpenVINO/whisper-medium-int8-ov/resolve/main/"
      },
      {
         "Whisper Medium (INT4)",
         "whisper-medium-int4-ov",
         "https://huggingface.co/OpenVINO/whisper-medium-int4-ov/resolve/main/"
      },
      {
         "Whisper Large V2 (FP16)",
         "whisper-large-v2-fp16-ov",
         "" //Not yet on HF. Hopefully soon!
      },
      {
         "Whisper Large V2 (INT8)",
         "whisper-large-v2-int8-ov",
         "" //Not yet on HF. Hopefully soon!
      },
      {
         "Whisper Large V2 (INT4)",
         "whisper-large-v2-int4-ov",
         "" //Not yet on HF. Hopefully soon!
      },
      {
         "Whisper Large V3 (FP16)",
         "whisper-large-v3-fp16-ov",
         "https://huggingface.co/OpenVINO/whisper-large-v3-fp16-ov/resolve/main/"
      },
      {
         "Whisper Large V3 (INT8)",
         "whisper-large-v3-int8-ov",
         "https://huggingface.co/OpenVINO/whisper-large-v3-int8-ov/resolve/main/"
      },
      {
         "Whisper Large V3 (INT4)",
         "whisper-large-v3-int4-ov",
         "https://huggingface.co/OpenVINO/whisper-large-v3-int4-ov/resolve/main/"
      },
      {
         "Whisper Large V3 Turbo (FP16)",
         "whisper-large-v3-turbo-fp16-ov",
         "" //Not yet on HF. Hopefully soon!
      },
      {
         "Whisper Large V3 Turbo (INT8)",
         "whisper-large-v3-turbo-int8-ov",
         "" //Not yet on HF. Hopefully soon!
      },
      {
         "Whisper Large V3 Turbo (INT4)",
         "whisper-large-v3-turbo-int4-ov",
         "" //Not yet on HF. Hopefully soon!
      },
      {
         "Distil-Whisper Base (FP16)",
         "distil-whisper-base-fp16-ov",
         "" // Distil-Base models on HF are kind of broken... they don't include the _with_past models
      },
      {
         "Distil-Whisper Base (INT8)",
         "distil-whisper-base-int8-ov",
         "" // Distil-Base models on HF are kind of broken... they don't include the _with_past models
      },
      {
         "Distil-Whisper Base (INT4)",
         "distil-whisper-base-int4-ov",
         "" // Distil-Base models on HF are kind of broken... they don't include the _with_past models
      },
      {
         "Distil-Whisper Medium (FP16)",
         "distil-whisper-medium-fp16-ov",
         "" // Distil-Medium models on HF are kind of broken... they don't include the _with_past models
      },
      {
         "Distil-Whisper Medium (INT8)",
         "distil-whisper-medium-int8-ov",
         "" // Distil-Medium models on HF are kind of broken... they don't include the _with_past models
      },
      {
         "Distil-Whisper Medium (INT4)",
         "distil-whisper-medium-int4-ov",
         "" // Distil-Medium models on HF are kind of broken... they don't include the _with_past models
      },
      {
         "Distil-Whisper Large V3 (FP16)",
         "distil-whisper-large-v3-fp16-ov",
         "https://huggingface.co/OpenVINO/distil-whisper-large-v3-fp16-ov/resolve/main/"
      },
      {
         "Distil-Whisper Large V3 (INT8)",
         "distil-whisper-large-v3-int8-ov",
         "https://huggingface.co/OpenVINO/distil-whisper-large-v3-int8-ov/resolve/main/"
      },
      {
         "Distil-Whisper Large V3 (INT4)",
         "distil-whisper-large-v3-int4-ov",
         "https://huggingface.co/OpenVINO/distil-whisper-large-v3-int4-ov/resolve/main/"
      },
   };

   auto whisper_collection = std::make_shared< OVModelManager::ModelCollection >();

   for (auto& whisper_model_info : whisper_model_info)
   {
      std::shared_ptr<OVModelManager::ModelInfo> whisper_info = std::make_shared<OVModelManager::ModelInfo>();
      whisper_info->model_name = whisper_model_info.ui_name;

      //TODO: add some more detailed info here..
      whisper_info->info = whisper_model_info.ui_name + " TODO: Add more info!";

      whisper_info->baseUrl = whisper_model_info.base_url;
      whisper_info->relative_path = "whisper/" + whisper_model_info.relative_path;
      whisper_info->fileList =
      {
         "added_tokens.json", "config.json", "generation_config.json", "normalizer.json", "openvino_decoder_model.bin",
         "openvino_decoder_model.xml", "openvino_detokenizer.bin", "openvino_detokenizer.xml", "openvino_encoder_model.bin", "openvino_encoder_model.xml",
         "openvino_tokenizer.bin", "openvino_tokenizer.xml", "preprocessor_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.json"
      };

      //TODO: This should go away once HF is updated with the latest conversions from optimum-cli.
      whisper_info->fileList.push_back("openvino_decoder_with_past_model.bin");
      whisper_info->fileList.push_back("openvino_decoder_with_past_model.xml");

      whisper_collection->models.push_back(whisper_info);
   }

   return whisper_collection;
}


void OVModelManager::_populate_model_collection()
{
   mModelCollection.insert({ MusicSepName(), populate_music_separation() });
   mModelCollection.insert({ MusicGenName(), populate_music_generation() });
   mModelCollection.insert({ WhisperName(), populate_whisper() });
}
