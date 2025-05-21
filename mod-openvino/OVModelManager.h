#pragma once
#include <unordered_map>
#include <string>
#include <memory>
#include <FileNames.h>

class OVModelManager {

public:

   struct ModelInfo
   {
      // The name of the model as displayed by the UI.
      std::string model_name;

      // The information that pops up when user clicks 'info' on the UI.
      std::string info;

      // The 'base' URL where each of the files in 'fileList' can be downloaded from. 
      std::string baseUrl;

      // The complete URL for each file is generated as:
      // baseUrl + filename + postUrl
      std::string postUrl = "?download=true";

      // relative folder path (away from from 'base' openvino-models folder).
      std::string relative_path;

      // List of file names expected to be present / downloaded.
      std::vector< std::string > fileList;

      // If true, all files in 'fileList' are present.
      bool installed = false;

      //This will be set to absolute path of openvino-models + relative_path, but only
      // if 'installed' is true. 
      std::string installation_path;                      
   };

   struct ModelCollection
   {
      std::vector< std::shared_ptr<ModelInfo> > models;
   };

   // strings to be passed into various functions below that take 'effect' as parameter.
   static const std::string MusicGenName() { return "Music Generation"; }
   static const std::string MusicSepName() { return "Music Separation"; };
   static const std::string NoiseSuppressName() { return "Noise Suppression"; }
   static const std::string SuperResName() { return "Super Resolution"; }
   static const std::string WhisperName() { return "Whisper Transcription"; }

   static OVModelManager& instance() {
      static OVModelManager instance;  
      return instance;
   }

   std::shared_ptr<ModelCollection> GetModelCollection(const std::string& effect);

   using ProgressCallback = std::function<void(float)>;
   void install_model(std::string effect, std::string model_id, ProgressCallback callback = nullptr);
   size_t install_model_size(std::string effect, std::string model_id);

   OVModelManager(const OVModelManager&) = delete;
   OVModelManager& operator=(const OVModelManager&) = delete;

private:
   OVModelManager();
   ~OVModelManager() = default;

   struct ModelCollection;
   std::unordered_map< std::string, std::shared_ptr<ModelCollection> > mModelCollection;

   std::vector< FilePath > mSearchPaths;

   void _check_installed_model(std::shared_ptr<ModelInfo> model_info);
   void _check_installed_models();

   // called once during construction.
   void _populate_model_collection();
};
