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
   mModelCollection.insert({ MusicSepName(), populate_music_separation()});
   mModelCollection.insert({ WhisperName(), populate_whisper() });
}
