#include "OVModelManager.h"
#include <NetworkManager.h>
#include <Request.h>
#include <IResponse.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <sstream>
#include <future>

#include <wx/file.h>


std::shared_ptr<OVModelManager::ModelCollection> OVModelManager::GetModelCollection(const std::string& effect)
{
   auto it = mModelCollection.find(effect);
   if (it == mModelCollection.end())
   {
      // If effect not in the map, return empty model collection
      return std::make_shared<ModelCollection>();
   }

   return it->second;
}

OVModelManager::OVModelManager()
{
   // Populate search paths where we look for installed models. Note that the first one appended to the list (appdata) is the *preferred*
   // location to find models, and this is where models will be installed to with the UI-based ModelManager.
   {
      FilePath appdata_openvino_models_path = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-models")).GetFullPath());
      mSearchPaths.push_back(appdata_openvino_models_path);

      //Add legacy search path so that we find models that were installed with older versions (from installer)
      FilePath legacy_openvino_models_path = wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath();
      mSearchPaths.push_back(legacy_openvino_models_path);
   }

   // initialize all of the details for all supported models.
   _populate_model_collection();

   // check which models are currently installed.
   _check_installed_models();
}

static inline std::vector<std::string> splitPath(const std::string& path, char delimiter = '/') {
   std::vector<std::string> parts;
   std::stringstream ss(path);
   std::string item;
   while (std::getline(ss, item, delimiter)) {
      if (!item.empty()) {
         parts.push_back(item);
      }
   }
   return parts;
}

void OVModelManager::_check_installed_model(std::shared_ptr<ModelInfo> model_info)
{
   for (auto& search_path_base : mSearchPaths)
   {
      bool all_found = true;
      for (auto& file : model_info->fileList)
      {
         wxFileName fullFilePath(search_path_base + "/" + model_info->relative_path + "/" + file);
         fullFilePath.Normalize();

         std::cout << "fullFilePath = " << fullFilePath.GetFullPath().ToStdString() << std::endl;
         if (!fullFilePath.FileExists())
         {
            all_found = false;
            std::cout << "    file doesn't exist." << std::endl;
            break;
         }
         else
         {
            std::cout << "    file exists." << std::endl;
         }
      }

      if (all_found)
      {
         auto split_path = splitPath(model_info->relative_path);
         auto fullInstallationPath = search_path_base;
         for (int i = 0; i < split_path.size(); i++)
         {
            fullInstallationPath = wxFileName(fullInstallationPath, wxString(split_path[i])).GetFullPath();
         }

         model_info->installed = true;
         model_info->installation_path = fullInstallationPath.ToStdString();
         std::cout << "Set installation path to " << model_info->installation_path << std::endl;
         break;
      }
   }
}

void OVModelManager::_check_installed_models()
{
   for (auto& collection_pair : mModelCollection)
   {
      auto& collection = collection_pair.second;
      for (auto& model_info : collection->models)
      {
         _check_installed_model(model_info);  
      }
   }
}

static inline void mkdir_relative_paths(std::string relative_file, wxString base_path){
   auto split_path = splitPath(relative_file);

   for (int i = 0; i < split_path.size() - 1; i++)
   {
      base_path = FileNames::MkDir(wxFileName(base_path, wxString(split_path[i])).GetFullPath());
   }
}

size_t OVModelManager::install_model_size(std::string effect, std::string model_id)
{
   auto it = mModelCollection.find(effect);
   if (it == mModelCollection.end()) {
      std::cout << "Model Collection for effect=" << effect << " not found." << std::endl;
      return 0;
   }

   std::shared_ptr<ModelInfo> model_info;
   auto collection = it->second;
   bool bFound = false;
   for (auto& info : collection->models) {
      if (info->model_name == model_id) {
         model_info = info;
         bFound = true;
      }
   }

   if (!bFound) {
      std::cout << "Model Info for model_id=" << model_id << " not found." << std::endl;
      return 0;
   }

   size_t total_size = 0;
   auto baseUrl = model_info->baseUrl;
   audacity::network_manager::NetworkManager& manager = audacity::network_manager::NetworkManager::GetInstance();

   for (auto& file : model_info->fileList) {
      std::string url = baseUrl + file + "?download=true";
      audacity::network_manager::Request request(url);
      auto response = manager.doHead(request);

      while (!response->isFinished())
      {
         std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

      if ((response->getHTTPCode() != 200) && (response->getHTTPCode() != 302)) {
         std::cout << "error fetching head for URL = " << url << std::endl;
         total_size = 0;
         break;
      }

#if 0
      std::cout << "file = " << file << std::endl;
      auto headers_list = response->getHeaders();
      for (auto& header : headers_list)
      {
         std::cout << "    header name: " << header.Name << std::endl;
         std::cout << "    header value: " << header.Value << std::endl;
      }
#endif

      // For LFS files on GitHub (usually large ones, like .bin's) have 'X-Linked-Size' headers,
      // so we look for those first. If that doesn't exist, we use 'Content-Length' header.
      std::vector < std::string> size_headers = { "X-Linked-Size", "Content-Length" };
      bool size_header_found = false;
      for (auto& header : size_headers)
      {
         if (response->hasHeader(header)) {
            std::string length = response->getHeader(header);
            int size = std::stoi(length);

            std::cout << "file: " << file << ", size = " << size << std::endl;
            total_size += size;

            size_header_found = true;
         }
      }

      if (!size_header_found)
      {
         std::cout << "response does not have 'X-Linked-Size' or 'Content-Length' headers for URL=" << url << std::endl;
         total_size = 0;
         break;
      }
   }

   return total_size;
}

void OVModelManager::install_model(std::string effect, std::string model_id, ProgressCallback callback)
{
   auto it = mModelCollection.find(effect);
   if (it == mModelCollection.end()) {
      std::cout << "Model Collection for effect=" << effect << " not found." << std::endl;
      return;
   }

   std::shared_ptr<ModelInfo> model_info;
   auto collection = it->second;
   bool bFound = false;
   for (auto& info : collection->models) {
      if (info->model_name == model_id) {
         model_info = info;
         bFound = true;
      }
   }

   if (!bFound) {
      std::cout << "Model Info for model_id=" << model_id << " not found." << std::endl;
      return;
   }

   //TODO: Replace this with a hard-coded size... I think.
   size_t total_download_size = install_model_size(effect, model_id);
   if (total_download_size == 0)
   {
      std::cout << "install_model_size failed." << std::endl;
      return;
   }
   size_t bytes_downloaded_so_far = 0;

   std::cout << "Total size we are about to download = " << total_download_size << std::endl;

   auto base_openvino_models_path = mSearchPaths[0];

   audacity::network_manager::NetworkManager& manager = audacity::network_manager::NetworkManager::GetInstance();

   bool bError = false;

   auto baseUrl = model_info->baseUrl;
   auto postUrl = model_info->postUrl;
   for (auto& file : model_info->fileList) {
      std::string url = baseUrl + file + postUrl;
      std::cout << "Downloading from URL: " << url << std::endl;
      audacity::network_manager::Request request(url);
      auto response = manager.doGet(request);

      mkdir_relative_paths(model_info->relative_path + "/" + file, base_openvino_models_path);
      wxFileName fullFilePath(base_openvino_models_path + "/" + model_info->relative_path + "/" + file);
      fullFilePath.Normalize();

      std::cout << "Saving to " << fullFilePath.GetFullPath().ToStdString() << std::endl;

      std::shared_ptr<wxFile> wx_file = std::make_shared<wxFile>(fullFilePath.GetFullPath(), wxFile::write);

      // write to file here
      response->setOnDataReceivedCallback(
         [this, response, wx_file, &bError , &bytes_downloaded_so_far, callback, &total_download_size](audacity::network_manager::IResponse*)
         {
            // only attempt save if request succeeded
            int httpCode = response->getHTTPCode();
            if ((httpCode == 200) || (httpCode == 302))
            {
               const std::string responseData = response->readAll<std::string>();
               size_t bytesWritten = wx_file->Write(responseData.c_str(), responseData.size());

               if (wx_file->Error()) {
                  int last_error = wx_file->GetLastError();

                  std::cout << "uh oh... ex_file Error! last_error=" << last_error << std::endl;
                  bError = true;
                  response->Cancel();
                  return;
               }

               bytes_downloaded_so_far += bytesWritten;

#if 0
               if (bytes_downloaded_so_far >= 1024 * 1024) {
                  bError = true;
                  std::cout << "fake error!" << std::endl;
                  response->Cancel();
                  return;
               }
#endif

               if (total_download_size > 0 && callback ) {
                  double perc_complete = static_cast<double>(bytes_downloaded_so_far) / static_cast<double>(total_download_size);
                  std::cout << "making callback with perc_complete=" << perc_complete << std::endl;
                  callback(static_cast<float>(perc_complete));
               }

               if (bytesWritten != responseData.size())
               {
                  std::cout << "uh oh... bytesWritten != responseData.size() " << std::endl;
                  bError = true;
                  response->Cancel();
                  return;
               }
            }
            else
            {
               std::cout << "uh oh... httpCode = " << httpCode << std::endl;
               bError = true;
               response->Cancel();
               return;
            }
         }
      );

      std::promise<void> donePromise;
      std::future<void> doneFuture = donePromise.get_future();

      response->setRequestFinishedCallback(
         [this, &donePromise](audacity::network_manager::IResponse*)
         {
            donePromise.set_value();
         }
      );

      //wait for request to complete.
      doneFuture.get();

      if (bError)
         break;

      std::cout << "finished downloading " << url << std::endl;
   }

   //re-run file check for this model.
   _check_installed_model(model_info);

   if (model_info->installed) {
      auto callback_it = mInstallCallbacks.find(effect);
      if (callback_it != mInstallCallbacks.end())
      {
         callback_it->second(model_info->model_name);
      }
   }
}

void OVModelManager::register_installed_callback(const std::string& effect, InstalledCallback callback)
{
   // For the 2nd+ register call, don't insert it.
   if (mInstallCallbacks.count(effect) == 0) {
      mInstallCallbacks.insert({ effect, callback });
   }
}
