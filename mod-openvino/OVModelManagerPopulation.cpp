#include "OVModelManager.h"
#include "model_md_card_info.h"

static std::shared_ptr< OVModelManager::ModelCollection > populate_music_separation()
{
   auto music_sep_collection = std::make_shared< OVModelManager::ModelCollection >();

   std::string relative_path = "stem_separation/";

   // demucs models
   {
      std::string demucs_baseURL = "https://huggingface.co/Intel/demucs-openvino/resolve/3e9e7d2f15c1ff4877917a224f2f9668c9c41881/";
      std::vector< std::string > fileList = { "htdemucs_fwd.bin", "htdemucs_fwd.xml" };
      {
         std::shared_ptr<OVModelManager::ModelInfo> demucs_model_info = std::make_shared<OVModelManager::ModelInfo>();
         demucs_model_info->model_name = "Demucs v4";
         demucs_model_info->info = music_separation_demucs_v4;
         demucs_model_info->baseUrl = demucs_baseURL + "htdemucs_v4/";
         demucs_model_info->relative_path = relative_path + "htdemucs_v4";
         demucs_model_info->fileList = fileList;
         music_sep_collection->models.emplace_back(demucs_model_info);
      }

      {
         std::shared_ptr<OVModelManager::ModelInfo> demucs_model_info = std::make_shared<OVModelManager::ModelInfo>();
         demucs_model_info->model_name = "Demucs v4 FT Drums";
         demucs_model_info->info = music_separation_demucs_v4_ft_drums;
         demucs_model_info->baseUrl = demucs_baseURL + "htdemucs_v4_ht_drums/";
         demucs_model_info->relative_path = relative_path + "htdemucs_v4_ht_drums";
         demucs_model_info->fileList = fileList;
         music_sep_collection->models.emplace_back(demucs_model_info);
      }

      {
         std::shared_ptr<OVModelManager::ModelInfo> demucs_model_info = std::make_shared<OVModelManager::ModelInfo>();
         demucs_model_info->model_name = "Demucs v4 FT Bass";
         demucs_model_info->info = music_separation_demucs_v4_ft_bass;
         demucs_model_info->baseUrl = demucs_baseURL + "htdemucs_v4_ht_bass/";
         demucs_model_info->relative_path = relative_path + "htdemucs_v4_ht_bass";
         demucs_model_info->fileList = fileList;
         music_sep_collection->models.emplace_back(demucs_model_info);
      }

      {
         std::shared_ptr<OVModelManager::ModelInfo> demucs_model_info = std::make_shared<OVModelManager::ModelInfo>();
         demucs_model_info->model_name = "Demucs v4 FT Other Instruments";
         demucs_model_info->info = music_separation_demucs_v4_ft_other;
         demucs_model_info->baseUrl = demucs_baseURL + "htdemucs_v4_ht_other/";
         demucs_model_info->relative_path = relative_path + "htdemucs_v4_ht_other";
         demucs_model_info->fileList = fileList;
         music_sep_collection->models.emplace_back(demucs_model_info);
      }

      {
         std::shared_ptr<OVModelManager::ModelInfo> demucs_model_info = std::make_shared<OVModelManager::ModelInfo>();
         demucs_model_info->model_name = "Demucs v4 FT Vocals";
         demucs_model_info->info = music_separation_demucs_v4_ft_vocals;
         demucs_model_info->baseUrl = demucs_baseURL + "htdemucs_v4_ht_vocals/";
         demucs_model_info->relative_path = relative_path + "htdemucs_v4_ht_vocals";
         demucs_model_info->fileList = fileList;
         music_sep_collection->models.emplace_back(demucs_model_info);
      }

      {
         std::shared_ptr<OVModelManager::ModelInfo> demucs_model_info = std::make_shared<OVModelManager::ModelInfo>();
         demucs_model_info->model_name = "Demucs v4 6s";
         demucs_model_info->info = music_separation_demucs_v4_6s;
         demucs_model_info->baseUrl = demucs_baseURL + "htdemucs_v4_6s/";
         demucs_model_info->relative_path = relative_path + "htdemucs_v4_6s";
         demucs_model_info->fileList = fileList;
         music_sep_collection->models.emplace_back(demucs_model_info);
      }
   }

   //mel band roformer models
   {
      std::vector< std::string > fileList = { "mel_band_pre.xml", "mel_band_pre.bin",
                                              "mel_band_post.xml", "mel_band_post.bin",
                                              "mel_band_fwd.xml", "mel_band_fwd.bin" };

      {
         std::shared_ptr<OVModelManager::ModelInfo> mel_model_info = std::make_shared<OVModelManager::ModelInfo>();
         mel_model_info->model_name = "MelBandRoformer Vocals (@KimberleyJensen)";
         mel_model_info->info = music_separation_mel_vocals_kimberley_jenson;
         mel_model_info->baseUrl = "https://huggingface.co/Intel/vocals_mel_band_roformer_kimberleyJSN_openvino/resolve/ce2bae0e27f9b115f38b1ddad35439df2d28cbbd/";
         mel_model_info->relative_path = relative_path + "melband_roformer_kimberley_jenson";
         mel_model_info->fileList = fileList;
         music_sep_collection->models.emplace_back(mel_model_info);
      }

      {
         std::shared_ptr<OVModelManager::ModelInfo> mel_model_info = std::make_shared<OVModelManager::ModelInfo>();
         mel_model_info->model_name = "MelBandRoformer Crowd (@aufr33, @viperx)";
         mel_model_info->info = music_separation_mel_crowd_aufr33_viperx;
         mel_model_info->baseUrl = "https://huggingface.co/Intel/crowd_mel_band_roformer_aufr33_viperx_openvino/resolve/b35f0dc8e9ee507582bc93a6e2b52e0dba9eca93/";
         mel_model_info->relative_path = relative_path + "melband_roformer_crowd";
         mel_model_info->fileList = fileList;
         music_sep_collection->models.emplace_back(mel_model_info);
      }
   }

   //MDX23C models
   {
      std::vector< std::string > fileList = { "mdx23c_fwd.xml", "mdx23c_fwd.bin"};

      {
         std::shared_ptr<OVModelManager::ModelInfo> mdx_model_info = std::make_shared<OVModelManager::ModelInfo>();
         mdx_model_info->model_name = "MDX23C Drum Separation (@jarredou)";
         mdx_model_info->info = music_separation_msdx23c_drum_sep_jarredou;
         mdx_model_info->baseUrl = "https://huggingface.co/Intel/drumsep_mdx23c_jarredou_openvino/resolve/2944425500506842ccc4ca130b22be8cfe95b20d/";
         mdx_model_info->relative_path = relative_path + "drumsep_jarredou_mdx23c";
         mdx_model_info->fileList = fileList;
         music_sep_collection->models.emplace_back(mdx_model_info);
      }
   }

   return music_sep_collection;
}


static std::shared_ptr< OVModelManager::ModelCollection > populate_reverb_removal()
{
   auto collection = std::make_shared< OVModelManager::ModelCollection >();

   std::string relative_path = "reverb_removal/";

   //mel band roformer models
   {
      std::vector< std::string > fileList = { "mel_band_pre.xml", "mel_band_pre.bin",
                                              "mel_band_post.xml", "mel_band_post.bin",
                                              "mel_band_fwd.xml", "mel_band_fwd.bin" };

      {
         std::shared_ptr<OVModelManager::ModelInfo> mel_model_info = std::make_shared<OVModelManager::ModelInfo>();
         mel_model_info->model_name = "MelBandRoformer Dereverb Mono (@anvuew)";
         mel_model_info->info = reverb_removal_mel_band_dereverb_mono_anvuew;
         mel_model_info->baseUrl = "https://huggingface.co/Intel/dereverb_mel_band_roformer_anvuew_openvino/resolve/16aeb6904702657415c04bdc906dc9c3ed6524a1/mono/";
         mel_model_info->relative_path = relative_path + "mel_band_roformer_mono_anvuew";
         mel_model_info->fileList = fileList;
         collection->models.emplace_back(mel_model_info);
      }
   }

   return collection;
}

static std::shared_ptr< OVModelManager::ModelCollection > populate_music_restoration()
{
   auto collection = std::make_shared< OVModelManager::ModelCollection >();

   std::string relative_path = "music_restoration/";

   //apollo models
   {
      std::vector< std::string > fileList = { "apollo_fwd.xml", "apollo_fwd.bin"};

      {
         std::shared_ptr<OVModelManager::ModelInfo> mel_model_info = std::make_shared<OVModelManager::ModelInfo>();
         mel_model_info->model_name = "Apollo MP3 Restore (@JusperLee)";
         mel_model_info->info = music_restoration_apollo_mp3_jusperlee;
         mel_model_info->baseUrl = "https://huggingface.co/Intel/apollo_jusperlee_openvino/resolve/720c90a7df79fd6add733ca9748a22b471a3bc09/";
         mel_model_info->relative_path = relative_path + "apollo_jusperlee";
         mel_model_info->fileList = fileList;
         collection->models.emplace_back(mel_model_info);
      }

      {
         std::shared_ptr<OVModelManager::ModelInfo> mel_model_info = std::make_shared<OVModelManager::ModelInfo>();
         mel_model_info->model_name = "Apollo Universal Restore (@Lew)";
         mel_model_info->info = music_restoration_apollo_universal_lew;
         mel_model_info->baseUrl = "";
         mel_model_info->relative_path = relative_path + "apollo_universal";
         mel_model_info->fileList = fileList;
         collection->models.emplace_back(mel_model_info);
      }
   }

   return collection;
}

struct WhisperInfo
{
   std::string ui_name;
   std::string relative_path;
   std::string base_url;
   std::string quick_description;
};

static std::shared_ptr< OVModelManager::ModelCollection > populate_whisper()
{
   const std::vector<WhisperInfo> whisper_model_infos
   {
      {
         "Whisper Base (FP16)",
         "whisper-base-fp16-ov",
         "https://huggingface.co/OpenVINO/whisper-base-fp16-ov/resolve/e95b28c093fc5f22d2b0d5b48524497d7784f308/",
         "FP16-quantized version of Whisper-Base. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Base (INT8)",
         "whisper-base-int8-ov",
         "https://huggingface.co/OpenVINO/whisper-base-int8-ov/resolve/ddb022a4299a78a0104e1f5b1eb2aae13859fc74/",
         "INT8-quantized version of Whisper-Base. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Base (INT4)",
         "whisper-base-int4-ov",
         "https://huggingface.co/OpenVINO/whisper-base-int4-ov/resolve/7d7a04e34adc1a1b7c9f14f0886daeed8900c892/",
         "INT4-quantized version of Whisper-Base. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Medium (FP16)",
         "whisper-medium-fp16-ov",
         "https://huggingface.co/OpenVINO/whisper-medium-fp16-ov/resolve/f44696c80386be16a024c91b3d75884367881ef2/",
         "FP16-quantized version of Whisper-Medium. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Medium (INT8)",
         "whisper-medium-int8-ov",
         "https://huggingface.co/OpenVINO/whisper-medium-int8-ov/resolve/32f273ed2b7b780171a0435b0922f27787d682d2/",
         "INT8-quantized version of Whisper-Medium. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Medium (INT4)",
         "whisper-medium-int4-ov",
         "https://huggingface.co/OpenVINO/whisper-medium-int4-ov/resolve/ff948b0e03fba6d41059225e8242a844b373dc76/",
         "INT4-quantized version of Whisper-Medium. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Large V2 (FP16)",
         "whisper-large-v2-fp16-ov",
         "", //Not yet on HF. Hopefully soon!
         "FP16-quantized version of Whisper-Large-V2. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Large V2 (INT8)",
         "whisper-large-v2-int8-ov",
         "", //Not yet on HF. Hopefully soon!
         "INT8-quantized version of Whisper-Large-V2. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Large V2 (INT4)",
         "whisper-large-v2-int4-ov",
         "", //Not yet on HF. Hopefully soon!
         "INT4-quantized version of Whisper-Large-V2. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Large V3 (FP16)",
         "whisper-large-v3-fp16-ov",
         "https://huggingface.co/OpenVINO/whisper-large-v3-fp16-ov/resolve/9e15f59e87f0b618f63c7da329bd77fcde5c26c1/",
         "FP16-quantized version of Whisper-Large-V3. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Large V3 (INT8)",
         "whisper-large-v3-int8-ov",
         "https://huggingface.co/OpenVINO/whisper-large-v3-int8-ov/resolve/b31e1dcee5de24d49c6cc96da2a603eae409e722/",
         "INT8-quantized version of Whisper-Large-V3. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Large V3 (INT4)",
         "whisper-large-v3-int4-ov",
         "https://huggingface.co/OpenVINO/whisper-large-v3-int4-ov/resolve/1c151299249b18003eabf874e02e2ed65bb08468/",
         "INT4-quantized version of Whisper-Large-V3. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Distil-Whisper Large V3 (FP16)",
         "distil-whisper-large-v3-fp16-ov",
         "https://huggingface.co/OpenVINO/distil-whisper-large-v3-fp16-ov/resolve/eef9b75180e7ff7a8fc026f6ef2cd6011de60fe7/",
         "FP16-quantized version of Distil-Whisper-Large-V3. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Distil-Whisper Large V3 (INT8)",
         "distil-whisper-large-v3-int8-ov",
         "https://huggingface.co/OpenVINO/distil-whisper-large-v3-int8-ov/resolve/81a5607b6139e8fbb7fb5aa73e9549323f1be258/",
         "INT8-quantized version of Distil-Whisper-Large-V3. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Distil-Whisper Large V3 (INT4)",
         "distil-whisper-large-v3-int4-ov",
         "https://huggingface.co/OpenVINO/distil-whisper-large-v3-int4-ov/resolve/d22aaac1a45a2dd82bf8485570927be06ff9d6ba/",
         "INT4-quantized version of Distil-Whisper-Large-V3. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Large V3 Turbo (FP16)",
         "whisper-large-v3-turbo-fp16-ov",
         "", //Not yet on HF. Hopefully soon!
         "FP16-quantized version of Whisper-Large-V3-Turbo. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Large V3 Turbo (INT8)",
         "whisper-large-v3-turbo-int8-ov",
         "", //Not yet on HF. Hopefully soon!
         "INT8-quantized version of Whisper-Large-V3-Turbo. See Quantization / Model Variant Guides below for more information."
      },
      {
         "Whisper Large V3 Turbo (INT4)",
         "whisper-large-v3-turbo-int4-ov",
         "", //Not yet on HF. Hopefully soon!
         "INT4-quantized version of Whisper-Large-V3-Turbo. See Quantization / Model Variant Guides below for more information."
      }
   };

   auto whisper_collection = std::make_shared< OVModelManager::ModelCollection >();

   for (auto& whisper_model_info : whisper_model_infos)
   {
      std::shared_ptr<OVModelManager::ModelInfo> whisper_info = std::make_shared<OVModelManager::ModelInfo>();
      whisper_info->model_name = whisper_model_info.ui_name;

      std::string info = "<h1>" + whisper_model_info.ui_name + "</h1>\n\n";
      info += "<p>" + whisper_model_info.quick_description + "</p>";

      // Add the 'general' info from whisper/info.md
      info += whisper_transcription_info;

      whisper_info->info = info;

      whisper_info->baseUrl = whisper_model_info.base_url;
      whisper_info->relative_path = "whisper/" + whisper_model_info.relative_path;
      whisper_info->fileList =
      {
         "added_tokens.json", "config.json", "generation_config.json", "normalizer.json", "openvino_decoder_model.bin",
         "openvino_decoder_model.xml", "openvino_detokenizer.bin", "openvino_detokenizer.xml", "openvino_encoder_model.bin", "openvino_encoder_model.xml",
         "openvino_tokenizer.bin", "openvino_tokenizer.xml", "preprocessor_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "vocab.json"
      };

      whisper_collection->models.push_back(whisper_info);
   }

   return whisper_collection;
}

static std::shared_ptr< OVModelManager::ModelCollection > populate_super_resolution()
{
   std::string baseUrl = "https://huggingface.co/Intel/versatile_audio_super_resolution_openvino/resolve/b98a5a9e21ede61cd556cd04d425bf5bfd675328/";
   std::shared_ptr<OVModelManager::ModelInfo> common = std::make_shared<OVModelManager::ModelInfo>();
   common->model_name = "Super Resolution Common";
   common->baseUrl = baseUrl;
   common->relative_path = "audiosr";
   common->fileList = { "audiosr_decoder.bin", "audiosr_decoder.xml", "audiosr_encoder.bin", "audiosr_encoder.xml",
                        "mel_24000_cpu.raw", "post_quant_conv.bin", "post_quant_conv.xml", "quant_conv.bin",
                        "quant_conv.xml", "vae_feature_extract.bin", "vae_feature_extract.xml", "vocoder.xml",
                        "vocoder.bin"};

   auto collection = std::make_shared< OVModelManager::ModelCollection >();

   //Basic (General) FP16
   {
      std::shared_ptr<OVModelManager::ModelInfo> model = std::make_shared<OVModelManager::ModelInfo>();
      model->model_name = "Basic (General) (FP16)";
      model->info = super_resolution_basic_general;
      model->baseUrl = baseUrl;
      model->relative_path = "audiosr";
      model->dependencies.push_back(common);
      model->fileList = { "basic/ddpm.xml", "basic/ddpm.bin" };
      collection->models.emplace_back(model);
   }

   //Speech FP16
   {
      std::shared_ptr<OVModelManager::ModelInfo> model = std::make_shared<OVModelManager::ModelInfo>();
      model->model_name = "Speech (FP16)";
      model->info = super_resolution_speech;
      model->baseUrl = baseUrl;
      model->relative_path = "audiosr";
      model->dependencies.push_back(common);
      model->fileList = { "speech/ddpm.xml", "speech/ddpm.bin" };
      collection->models.emplace_back(model);
   }

   return collection;
}

static std::shared_ptr< OVModelManager::ModelCollection > populate_noise_suppression()
{
   auto collection = std::make_shared< OVModelManager::ModelCollection >();

   //deepfilternet
   {
      std::string baseUrl = "https://huggingface.co/Intel/deepfilternet-openvino/resolve/0615a1b18be8585156130a98fdbca75e9719eda3/";

      // deepfilternet2
      {
         std::shared_ptr<OVModelManager::ModelInfo> model = std::make_shared<OVModelManager::ModelInfo>();
         model->model_name = "DeepFilterNet2";
         model->info = noise_suppression_deepfilternet2;
         model->baseUrl = baseUrl;
         model->fileList = { "df_dec.bin", "df_dec.xml", "enc.xml", "enc.bin", "erb_dec.xml", "erb_dec.bin" };

         for (auto& f : model->fileList) {
            f = "deepfilternet2/" + f;
         }

         collection->models.emplace_back(model);
      }

      // deepfilternet3
      {
         std::shared_ptr<OVModelManager::ModelInfo> model = std::make_shared<OVModelManager::ModelInfo>();
         model->model_name = "DeepFilterNet3";
         model->info = noise_suppression_deepfilternet3;
         model->baseUrl = baseUrl;
         model->fileList = { "df_dec.bin", "df_dec.xml", "enc.xml", "enc.bin", "erb_dec.xml", "erb_dec.bin" };

         for (auto& f : model->fileList) {
            f = "deepfilternet3/" + f;
         }

         collection->models.emplace_back(model);
      }
   }

   // denseunet
   {
      std::shared_ptr<OVModelManager::ModelInfo> model = std::make_shared<OVModelManager::ModelInfo>();
      model->model_name = "DenseUNet";
      model->info = noise_suppression_denseunet;
      model->baseUrl = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16/";
      model->postUrl = "";
      model->fileList = { "noise-suppression-denseunet-ll-0001.xml", "noise-suppression-denseunet-ll-0001.bin" };
      collection->models.emplace_back(model);
   }

   return collection;
}


void OVModelManager::_populate_model_collection()
{
   mModelCollection.insert({ MusicSepName(), populate_music_separation() });
   mModelCollection.insert({ NoiseSuppressName(), populate_noise_suppression()});
   mModelCollection.insert({ SuperResName(), populate_super_resolution() });
   mModelCollection.insert({ WhisperName(), populate_whisper() });
   mModelCollection.insert({ ReverbRemovalName(), populate_reverb_removal() });
   mModelCollection.insert({ MusicRestorationName(), populate_music_restoration() });
}
