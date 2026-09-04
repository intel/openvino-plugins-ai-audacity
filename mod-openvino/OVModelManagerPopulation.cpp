#include "OVModelManager.h"
#include "model_download_manifest_info.h"
#include "model_md_card_info.h"

namespace {

const char* ResolveModelInfoFromKey(const std::string& info_key)
{
   if (info_key == "music_separation_demucs_v4")
      return music_separation_demucs_v4;
   if (info_key == "music_separation_demucs_v4_ft_drums")
      return music_separation_demucs_v4_ft_drums;
   if (info_key == "music_separation_demucs_v4_ft_bass")
      return music_separation_demucs_v4_ft_bass;
   if (info_key == "music_separation_demucs_v4_ft_other")
      return music_separation_demucs_v4_ft_other;
   if (info_key == "music_separation_demucs_v4_ft_vocals")
      return music_separation_demucs_v4_ft_vocals;
   if (info_key == "music_separation_demucs_v4_6s")
      return music_separation_demucs_v4_6s;
   if (info_key == "music_separation_mel_vocals_kimberley_jenson")
      return music_separation_mel_vocals_kimberley_jenson;
   if (info_key == "music_separation_mel_crowd_aufr33_viperx")
      return music_separation_mel_crowd_aufr33_viperx;
   if (info_key == "music_separation_msdx23c_drum_sep_jarredou")
      return music_separation_msdx23c_drum_sep_jarredou;

   if (info_key == "reverb_removal_mel_band_dereverb_mono_anvuew")
      return reverb_removal_mel_band_dereverb_mono_anvuew;

   if (info_key == "music_restoration_apollo_mp3_jusperlee")
      return music_restoration_apollo_mp3_jusperlee;
   if (info_key == "music_restoration_apollo_universal_lew")
      return music_restoration_apollo_universal_lew;

   if (info_key == "noise_suppression_deepfilternet2")
      return noise_suppression_deepfilternet2;
   if (info_key == "noise_suppression_deepfilternet3")
      return noise_suppression_deepfilternet3;
   if (info_key == "noise_suppression_denseunet")
      return noise_suppression_denseunet;

   if (info_key == "super_resolution_basic_general")
      return super_resolution_basic_general;
   if (info_key == "super_resolution_speech")
      return super_resolution_speech;

   if (info_key == "text_to_speech_kokoro_82m")
      return text_to_speech_kokoro_82m;

   if (info_key == "whisper_transcription_info")
      return whisper_transcription_info;

   return "";
}

void CopyManifestModelMetadata(const model_download_manifest::ModelInfo& source_model,
   const std::shared_ptr<OVModelManager::ModelInfo>& model_info)
{
   if (!model_info) {
      return;
   }

   model_info->baseUrl = source_model.base_url;
      model_info->revision = source_model.revision;
   model_info->postUrl = source_model.post_url;
   model_info->relative_path = source_model.relative_path;

   model_info->files.clear();
   model_info->files.reserve(source_model.file_count);
   for (std::size_t file_index = 0; file_index < source_model.file_count; ++file_index) {
      const auto& source_file = source_model.files[file_index];
      model_info->files.push_back({ source_file.name, source_file.expected_sha256, source_file.expected_size });
   }
}

template<typename InfoResolver, typename VisibilityPredicate>
std::shared_ptr<OVModelManager::ModelCollection> BuildManifestCollection(
   const std::string& effect,
   InfoResolver&& resolveInfo,
   VisibilityPredicate&& isVisible)
{
   auto collection = std::make_shared<OVModelManager::ModelCollection>();
   std::unordered_map<std::string, std::shared_ptr<OVModelManager::ModelInfo>> models_by_id;

   for (std::size_t i = 0; i < model_download_manifest::kModelCount; ++i)
   {
      const auto& source_model = model_download_manifest::kModels[i];
      if (source_model.effect != effect) {
         continue;
      }

      auto model_info = std::make_shared<OVModelManager::ModelInfo>();
      model_info->model_name = source_model.model_name;
      model_info->info = resolveInfo(source_model);
      CopyManifestModelMetadata(source_model, model_info);

      models_by_id[source_model.model_id] = model_info;
      if (isVisible(source_model)) {
         collection->models.emplace_back(model_info);
      }
   }

   for (std::size_t i = 0; i < model_download_manifest::kModelCount; ++i)
   {
      const auto& source_model = model_download_manifest::kModels[i];
      if (source_model.effect != effect) {
         continue;
      }

      auto model_it = models_by_id.find(source_model.model_id);
      if (model_it == models_by_id.end()) {
         continue;
      }

      auto& dependencies = model_it->second->dependencies;
      for (std::size_t dependency_index = 0; dependency_index < source_model.dependency_count; ++dependency_index) {
         const std::string dependency_id = source_model.dependencies[dependency_index];
         auto dependency_it = models_by_id.find(dependency_id);
         if (dependency_it != models_by_id.end()) {
            dependencies.push_back(dependency_it->second);
         }
      }
   }

   return collection;
}

std::string WhisperQuickDescription(const std::string& model_id)
{
   static const std::unordered_map<std::string, std::string> quick_description_by_id {
      { "whisper_base_fp16", "FP16-quantized version of Whisper-Base. See Quantization / Model Variant Guides below for more information." },
      { "whisper_base_int8", "INT8-quantized version of Whisper-Base. See Quantization / Model Variant Guides below for more information." },
      { "whisper_base_int4", "INT4-quantized version of Whisper-Base. See Quantization / Model Variant Guides below for more information." },
      { "whisper_medium_fp16", "FP16-quantized version of Whisper-Medium. See Quantization / Model Variant Guides below for more information." },
      { "whisper_medium_int8", "INT8-quantized version of Whisper-Medium. See Quantization / Model Variant Guides below for more information." },
      { "whisper_medium_int4", "INT4-quantized version of Whisper-Medium. See Quantization / Model Variant Guides below for more information." },
      { "whisper_large_v2_fp16", "FP16-quantized version of Whisper-Large-V2. See Quantization / Model Variant Guides below for more information." },
      { "whisper_large_v2_int8", "INT8-quantized version of Whisper-Large-V2. See Quantization / Model Variant Guides below for more information." },
      { "whisper_large_v2_int4", "INT4-quantized version of Whisper-Large-V2. See Quantization / Model Variant Guides below for more information." },
      { "whisper_large_v3_fp16", "FP16-quantized version of Whisper-Large-V3. See Quantization / Model Variant Guides below for more information." },
      { "whisper_large_v3_int8", "INT8-quantized version of Whisper-Large-V3. See Quantization / Model Variant Guides below for more information." },
      { "whisper_large_v3_int4", "INT4-quantized version of Whisper-Large-V3. See Quantization / Model Variant Guides below for more information." },
      { "distil_whisper_large_v3_fp16", "FP16-quantized version of Distil-Whisper-Large-V3. See Quantization / Model Variant Guides below for more information." },
      { "distil_whisper_large_v3_int8", "INT8-quantized version of Distil-Whisper-Large-V3. See Quantization / Model Variant Guides below for more information." },
      { "distil_whisper_large_v3_int4", "INT4-quantized version of Distil-Whisper-Large-V3. See Quantization / Model Variant Guides below for more information." },
      { "whisper_large_v3_turbo_fp16", "FP16-quantized version of Whisper-Large-V3-Turbo. See Quantization / Model Variant Guides below for more information." },
      { "whisper_large_v3_turbo_int8", "INT8-quantized version of Whisper-Large-V3-Turbo. See Quantization / Model Variant Guides below for more information." },
      { "whisper_large_v3_turbo_int4", "INT4-quantized version of Whisper-Large-V3-Turbo. See Quantization / Model Variant Guides below for more information." },
   };

   auto it = quick_description_by_id.find(model_id);
   if (it != quick_description_by_id.end()) {
      return it->second;
   }

   return "See Quantization / Model Variant Guides below for more information.";
}

} // namespace

static std::shared_ptr< OVModelManager::ModelCollection > populate_music_separation()
{
   return BuildManifestCollection(
      OVModelManager::MusicSepName(),
      [](const model_download_manifest::ModelInfo& source_model) {
         return std::string(ResolveModelInfoFromKey(source_model.info_key));
      },
      [](const model_download_manifest::ModelInfo&) { return true; });
}

static std::shared_ptr< OVModelManager::ModelCollection > populate_reverb_removal()
{
   return BuildManifestCollection(
      OVModelManager::ReverbRemovalName(),
      [](const model_download_manifest::ModelInfo& source_model) {
         return std::string(ResolveModelInfoFromKey(source_model.info_key));
      },
      [](const model_download_manifest::ModelInfo&) { return true; });
}

static std::shared_ptr< OVModelManager::ModelCollection > populate_music_restoration()
{
   return BuildManifestCollection(
      OVModelManager::MusicRestorationName(),
      [](const model_download_manifest::ModelInfo& source_model) {
         return std::string(ResolveModelInfoFromKey(source_model.info_key));
      },
      [](const model_download_manifest::ModelInfo&) { return true; });
}

static std::shared_ptr< OVModelManager::ModelCollection > populate_whisper()
{
   return BuildManifestCollection(
      OVModelManager::WhisperName(),
      [](const model_download_manifest::ModelInfo& source_model) {
         std::string info = "<h1>" + std::string(source_model.model_name) + "</h1>\n\n";
         info += "<p>" + WhisperQuickDescription(source_model.model_id) + "</p>";
         info += whisper_transcription_info;
         return info;
      },
      [](const model_download_manifest::ModelInfo&) { return true; });
}

static std::shared_ptr< OVModelManager::ModelCollection > populate_super_resolution()
{
   return BuildManifestCollection(
      OVModelManager::SuperResName(),
      [](const model_download_manifest::ModelInfo& source_model) {
         return std::string(ResolveModelInfoFromKey(source_model.info_key));
      },
      [](const model_download_manifest::ModelInfo& source_model) {
         return std::string(source_model.model_id) != "super_resolution_common";
      });
}

static std::shared_ptr< OVModelManager::ModelCollection > populate_noise_suppression()
{
   return BuildManifestCollection(
      OVModelManager::NoiseSuppressName(),
      [](const model_download_manifest::ModelInfo& source_model) {
         return std::string(ResolveModelInfoFromKey(source_model.info_key));
      },
      [](const model_download_manifest::ModelInfo&) { return true; });
}

static std::shared_ptr< OVModelManager::ModelCollection > populate_tts()
{
   return BuildManifestCollection(
      OVModelManager::TtsName(),
      [](const model_download_manifest::ModelInfo& source_model) {
         return std::string(ResolveModelInfoFromKey(source_model.info_key));
      },
      [](const model_download_manifest::ModelInfo&) { return true; });
}

void OVModelManager::_populate_model_collection()
{
   mModelCollection.insert({ MusicSepName(), populate_music_separation() });
   mModelCollection.insert({ NoiseSuppressName(), populate_noise_suppression() });
   mModelCollection.insert({ SuperResName(), populate_super_resolution() });
   mModelCollection.insert({ WhisperName(), populate_whisper() });
   mModelCollection.insert({ ReverbRemovalName(), populate_reverb_removal() });
   mModelCollection.insert({ MusicRestorationName(), populate_music_restoration() });
   mModelCollection.insert({ TtsName(), populate_tts() });
}
