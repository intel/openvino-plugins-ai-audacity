#include "OVModelManager.h"
#include "OpenVINOPluginPrefs.h"
#ifdef HAS_NETWORKING
#include <NetworkManager.h>
#include <Request.h>
#include <IResponse.h>
#include <crypto/SHA256.h>
#endif
#include <thread>
#include <chrono>
#include <sstream>
#include <future>
#include <array>
#include <algorithm>
#include <cctype>

#include <wx/file.h>
#include <wx/log.h>

namespace {

constexpr size_t DownloadBufferSize = 64 * 1024;
constexpr int SizeLookupMaxAttempts = 3;
constexpr auto SizeLookupRetryDelay = std::chrono::milliseconds(250);
constexpr int DownloadMaxAttempts = 3;
constexpr auto DownloadRetryDelay = std::chrono::milliseconds(250);

std::string NormalizeHexDigest(std::string digest)
{
   std::transform(digest.begin(), digest.end(), digest.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
   return digest;
}

std::string NormalizePathForComparison(const wxString& path)
{
   auto normalized = path.ToStdString();
   std::transform(normalized.begin(), normalized.end(), normalized.begin(),
      [](unsigned char c) {
         if (c == '\\') {
            return '/';
         }
         return static_cast<char>(std::tolower(c));
      });
   return normalized;
}

bool IsPathWithinBase(const wxString& basePath, const wxString& candidatePath)
{
   auto normalizedBase = NormalizePathForComparison(basePath);
   auto normalizedCandidate = NormalizePathForComparison(candidatePath);

   if (!normalizedBase.empty() && normalizedBase.back() != '/') {
      normalizedBase.push_back('/');
   }

   return normalizedCandidate.rfind(normalizedBase, 0) == 0;
}

std::string BuildInstallDetails(
   const std::string& effect,
   const std::string& modelName,
   const std::string& stage,
   const std::string& message,
   const std::string& resource = {},
   const std::string& extra = {})
{
   std::ostringstream stream;
   stream << "Effect: " << effect << "\n";
   stream << "Model: " << modelName << "\n";
   stream << "Stage: " << stage << "\n";
   stream << "Message: " << message;

   if (!resource.empty()) {
      stream << "\nResource: " << resource;
   }

   if (!extra.empty()) {
      stream << "\nDetails: " << extra;
   }

   return stream.str();
}

OVModelManager::InstallResult WrapInstallFailure(
   const std::string& effect,
   const std::string& requestedModel,
   const std::string& stage,
   const std::string& message,
   const OVModelManager::InstallResult& cause,
   const std::string& resource = {})
{
   std::ostringstream extra;
   if (!cause.summary.empty()) {
      extra << "Cause Summary: " << cause.summary;
   }

   if (!cause.details.empty()) {
      if (extra.tellp() > 0) {
         extra << "\n\n";
      }
      extra << cause.details;
   }

   return OVModelManager::InstallResult::Failure(
      message,
      BuildInstallDetails(effect, requestedModel, stage, message, resource, extra.str()));
}

}


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
   auto model_install_path = FileNames::MkDir(wxFileName(OpenVINOPluginSettings::GetOrCreateModelDir(true)).GetFullPath());
   mSearchPaths.push_back(model_install_path);

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

static void _check_installed_model_impl(std::shared_ptr<OVModelManager::ModelInfo> model_info, const FilePath& search_path_base)
{
   model_info->installed = false;

   if (!model_info->dependencies.empty())
   {
      for (auto& d : model_info->dependencies) {
         _check_installed_model_impl(d, search_path_base);

         if (!d->installed)
         {
            // of the dependencies aren't installed, then no point in proceeding.
            return;
         }
      }
   }

   bool all_found = true;
   for (const auto& file : model_info->files)
   {
      wxFileName fullFilePath(search_path_base + "/" + model_info->relative_path + "/" + file.name);
      fullFilePath.Normalize();

      if (!fullFilePath.FileExists())
      {
         all_found = false;
         break;
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
   }
}

void OVModelManager::_check_installed_model(std::shared_ptr<ModelInfo> model_info)
{
   for (auto& search_path_base : mSearchPaths)
   {
      _check_installed_model_impl(model_info, search_path_base);
      if (model_info->installed)
         break;
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

OVModelManager::InstallResult OVModelManager::install_model_size(std::shared_ptr<ModelInfo> model_info, size_t& total_size)
{
   total_size = 0;
#ifdef HAS_NETWORKING
   if (!model_info) {
   wxLogError("OVModelManager::install_model_size called on null model_info.");
      return InstallResult::Failure(
         "Model size calculation failed.",
         BuildInstallDetails({}, {}, "Size Check", "install_model_size received a null model pointer."));
   }

   auto baseUrl = model_info->baseUrl;
   audacity::network_manager::NetworkManager& manager = audacity::network_manager::NetworkManager::GetInstance();

   for (const auto& file : model_info->files) {
      std::string url = baseUrl + file.name + "?download=true";
      audacity::network_manager::Request request;

      try {
         request = audacity::network_manager::Request(url);
      }
      catch (const std::exception& error) {
         wxLogError("OVModelManager: failed to create HEAD request for URL '%s'. Exception: %s", url.c_str(), error.what());
         return InstallResult::Failure(
            "Could not create a download request.",
            BuildInstallDetails({}, model_info->model_name, "Size Check", "Could not create a HEAD request for model download size lookup.", url, error.what()));
      }

      InstallResult lastFailure = InstallResult::Failure(
         "Could not retrieve model download metadata.",
         BuildInstallDetails({}, model_info->model_name, "Size Check", "Model size lookup did not run.", url));

      bool sizeResolved = false;
      for (int attempt = 1; attempt <= SizeLookupMaxAttempts; ++attempt) {
         try {
            auto response = manager.doHead(request);

            while (!response->isFinished())
            {
               std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            if ((response->getHTTPCode() != 200) && (response->getHTTPCode() != 302)) {
               const auto errorString = response->getErrorString();
               wxLogWarning("OVModelManager: HEAD metadata attempt %d/%d returned HTTP status %d for URL '%s'. Network error: %s",
                  attempt, SizeLookupMaxAttempts, response->getHTTPCode(), url.c_str(), errorString.c_str());
               lastFailure = InstallResult::Failure(
                  "Could not retrieve model download metadata.",
                  BuildInstallDetails({}, model_info->model_name, "Size Check", "HEAD request returned an unexpected HTTP status.", url,
                     "Attempt: " + std::to_string(attempt) + "/" + std::to_string(SizeLookupMaxAttempts)
                     + "\nHTTP status: " + std::to_string(response->getHTTPCode())
                     + (errorString.empty() ? std::string{} : "\nNetwork error: " + errorString)));
            }
            else {
               // For LFS files on GitHub (usually large ones, like .bin's) have 'X-Linked-Size' headers,
               // so we look for those first. If that doesn't exist, we use 'Content-Length' header.
               std::vector < std::string> size_headers = { "X-Linked-Size", "Content-Length" };
               bool size_header_found = false;
               for (auto& header : size_headers)
               {
                  if (response->hasHeader(header)) {
                     std::string length = response->getHeader(header);
                     size_t size = (size_t)std::stoull(length);
                     total_size += size;

                     size_header_found = true;
                     sizeResolved = true;
                     break;
                  }
               }

               if (size_header_found) {
                  break;
               }

               wxLogWarning("OVModelManager: HEAD metadata attempt %d/%d missing size headers for URL '%s'.",
                  attempt, SizeLookupMaxAttempts, url.c_str());
               lastFailure = InstallResult::Failure(
                  "Could not determine model download size.",
                  BuildInstallDetails({}, model_info->model_name, "Size Check", "Response did not include X-Linked-Size or Content-Length.", url,
                     "Attempt: " + std::to_string(attempt) + "/" + std::to_string(SizeLookupMaxAttempts)));
            }
         }
         catch (const std::exception& error) {
            wxLogWarning("OVModelManager: exception during HEAD metadata attempt %d/%d for URL '%s'. Exception: %s",
               attempt, SizeLookupMaxAttempts, url.c_str(), error.what());
            lastFailure = InstallResult::Failure(
               "Could not retrieve model download metadata.",
               BuildInstallDetails({}, model_info->model_name, "Size Check", "Exception while reading HEAD response for size calculation.", url,
                  "Attempt: " + std::to_string(attempt) + "/" + std::to_string(SizeLookupMaxAttempts) + "\n" + error.what()));
         }

         if (attempt < SizeLookupMaxAttempts) {
            std::this_thread::sleep_for(SizeLookupRetryDelay);
         }
      }

      if (!sizeResolved) {
         return lastFailure;
      }
   }

   return InstallResult::Success();
#else
   return InstallResult::Failure(
      "Model downloads are not available in this build.",
      BuildInstallDetails({}, model_info ? model_info->model_name : std::string{}, "Size Check", "install_model_size called, but this build has no networking support."));
#endif
}

static OVModelManager::InstallResult download_model_files(const std::string& effect, std::shared_ptr<OVModelManager::ModelInfo> model_info, const FilePath &base_openvino_models_path, size_t total_download_size,
   size_t& bytes_downloaded_so_far, OVModelManager::ProgressCallback callback)
{
#ifdef HAS_NETWORKING
   if (!model_info) {
      return OVModelManager::InstallResult::Failure(
         "Model download failed.",
         BuildInstallDetails(effect, {}, "Download", "download_model_files received a null model pointer."));
   }

   audacity::network_manager::NetworkManager& manager = audacity::network_manager::NetworkManager::GetInstance();

   bool bError = false;

   auto baseUrl = model_info->baseUrl;
   auto postUrl = model_info->postUrl;
   wxFileName baseModelsPath(base_openvino_models_path);
   baseModelsPath.Normalize();
   const auto normalizedBaseModelsPath = baseModelsPath.GetFullPath();

   for (const auto& file : model_info->files) {
      std::string url = baseUrl + file.name + postUrl;

      OVModelManager::InstallResult lastFailure = OVModelManager::InstallResult::Failure(
         "Model download failed.",
         BuildInstallDetails(effect, model_info->model_name, "Download", "Download attempt did not run.", url));
      bool fileDownloaded = false;
      const auto bytesDownloadedBeforeFile = bytes_downloaded_so_far;

      for (int attempt = 1; attempt <= DownloadMaxAttempts; ++attempt) {
         bError = false;
         bytes_downloaded_so_far = bytesDownloadedBeforeFile;

         audacity::network_manager::Request request;
         std::shared_ptr<audacity::network_manager::IResponse> response;
         try {
            request = audacity::network_manager::Request(url);
            response = manager.doGet(request);
         }
         catch (const std::exception& error) {
            lastFailure = OVModelManager::InstallResult::Failure(
               "Model download request failed.",
               BuildInstallDetails(effect, model_info->model_name, "Download", "Could not start a GET request for the model file.", url,
                  "Attempt: " + std::to_string(attempt) + "/" + std::to_string(DownloadMaxAttempts) + "\n" + error.what()));
            if (attempt < DownloadMaxAttempts) {
               std::this_thread::sleep_for(DownloadRetryDelay);
               continue;
            }
            return lastFailure;
         }

         mkdir_relative_paths(model_info->relative_path + "/" + file.name, base_openvino_models_path);
         wxFileName fullFilePath(base_openvino_models_path + "/" + model_info->relative_path + "/" + file.name);
         fullFilePath.Normalize();

         if (!IsPathWithinBase(normalizedBaseModelsPath, fullFilePath.GetFullPath())) {
            return OVModelManager::InstallResult::Failure(
               "Model path validation failed.",
               BuildInstallDetails(
                  effect,
                  model_info->model_name,
                  "Path Validation",
                  "Resolved model file path escaped the configured model directory.",
                  fullFilePath.GetFullPath().ToStdString(),
                  "Model base path: " + normalizedBaseModelsPath.ToStdString()));
         }

         const auto tempFilePath = fullFilePath.GetFullPath() + ".tmp";
         const auto oldFilePath = fullFilePath.GetFullPath() + ".old";

         if (wxFileExists(tempFilePath)) {
            wxRemoveFile(tempFilePath);
         }

         std::shared_ptr<wxFile> wx_file = std::make_shared<wxFile>(tempFilePath, wxFile::write);
         if (!wx_file->IsOpened()) {
            return OVModelManager::InstallResult::Failure(
               "Could not open the destination file for writing.",
               BuildInstallDetails(effect, model_info->model_name, "Download", "Failed to open the destination temp file before writing downloaded data.", tempFilePath.ToStdString()));
         }

         std::string file_error_summary;
         std::string file_error_details;
         auto downloadBuffer = std::make_shared<std::array<uint8_t, DownloadBufferSize>>();
         auto fileHasher = std::make_shared<crypto::SHA256>();

         response->setOnDataReceivedCallback(
            [response, wx_file, downloadBuffer, fileHasher, tempFilePath, &bError, &bytes_downloaded_so_far, callback, &total_download_size, &file_error_summary, &file_error_details, &effect, model_info, url, fullFilePath](audacity::network_manager::IResponse*)
            {
               int httpCode = response->getHTTPCode();
               if ((httpCode == 200) || (httpCode == 302))
               {
                  while (true)
                  {
                     const auto bytesRead = response->readData(downloadBuffer->data(), downloadBuffer->size());
                     if (bytesRead == 0) {
                        break;
                     }

                     const size_t bytesWritten = wx_file->Write(downloadBuffer->data(), bytesRead);

                     if (wx_file->Error()) {
                        int last_error = wx_file->GetLastError();

                        wxLogError("OVModelManager: file write error (wxFile last_error=%d) for '%s'.", last_error, fullFilePath.GetFullPath());
                        file_error_summary = "Writing downloaded model data failed.";
                        file_error_details = BuildInstallDetails(
                           effect,
                           model_info->model_name,
                           "Download",
                           "wxFile reported an error while writing the downloaded data.",
                           tempFilePath.ToStdString(),
                           "wxFile last error: " + std::to_string(last_error) + "\nSource URL: " + url);
                        bError = true;
                        response->Cancel();
                        return;
                     }

                     bytes_downloaded_so_far += bytesWritten;
                     fileHasher->Update(downloadBuffer->data(), static_cast<std::size_t>(bytesRead));

                     if (total_download_size > 0 && callback) {
                        double perc_complete = static_cast<double>(bytes_downloaded_so_far) / static_cast<double>(total_download_size);
                        callback(static_cast<float>(perc_complete));
                     }

                     if (bytesWritten != bytesRead)
                     {
                        wxLogError("OVModelManager: incomplete file write for '%s' (written=%llu, received=%llu).",
                           fullFilePath.GetFullPath(),
                           static_cast<unsigned long long>(bytesWritten),
                           static_cast<unsigned long long>(bytesRead));
                        file_error_summary = "Incomplete model file write detected.";
                        file_error_details = BuildInstallDetails(
                           effect,
                           model_info->model_name,
                           "Download",
                           "The downloaded data size did not match the number of bytes written to disk.",
                           tempFilePath.ToStdString(),
                           "Bytes written: " + std::to_string(bytesWritten) + "\nBytes received: " + std::to_string(bytesRead) + "\nSource URL: " + url);
                        bError = true;
                        response->Cancel();
                        return;
                     }
                  }
               }
               else
               {
                  wxLogError("OVModelManager: GET request returned unexpected HTTP status %d for URL '%s'.", httpCode, url.c_str());
                  file_error_summary = "Model download returned an unexpected HTTP status.";
                  file_error_details = BuildInstallDetails(
                     effect,
                     model_info->model_name,
                     "Download",
                     "GET request returned an unexpected HTTP status.",
                     url,
                     "HTTP status: " + std::to_string(httpCode));
                  bError = true;
                  response->Cancel();
                  return;
               }
            }
         );

         std::promise<void> donePromise;
         std::future<void> doneFuture = donePromise.get_future();

         response->setRequestFinishedCallback(
            [&donePromise](audacity::network_manager::IResponse*)
            {
               donePromise.set_value();
            }
         );

         doneFuture.get();

         const auto networkError = response->getError();
         const auto networkErrorString = response->getErrorString();
         if (!bError && networkError != audacity::network_manager::NetworkError::NoError) {
            file_error_summary = "Model download failed due to a network error.";
            file_error_details = BuildInstallDetails(
               effect,
               model_info->model_name,
               "Download",
               "The GET request finished with a network error.",
               url,
               "Attempt: " + std::to_string(attempt) + "/" + std::to_string(DownloadMaxAttempts)
               + (networkErrorString.empty() ? std::string{} : "\nNetwork error: " + networkErrorString));
            bError = true;
         }

         if (bError) {
            if (wx_file->IsOpened()) {
               wx_file->Close();
            }
            if (wxFileExists(tempFilePath)) {
               wxRemoveFile(tempFilePath);
            }

            lastFailure = OVModelManager::InstallResult::Failure(file_error_summary, file_error_details);
            if (attempt < DownloadMaxAttempts) {
               wxLogWarning("OVModelManager: retrying download attempt %d/%d for '%s'.", attempt + 1, DownloadMaxAttempts, url.c_str());
               std::this_thread::sleep_for(DownloadRetryDelay);
               continue;
            }
            return lastFailure;
         }

         const auto fileHash = fileHasher->Finalize();
         if (!file.expected_sha256.empty()) {
            const auto normalizedExpected = NormalizeHexDigest(file.expected_sha256);
            const auto normalizedActual = NormalizeHexDigest(fileHash);
            if (normalizedExpected != normalizedActual) {
               wxLogWarning("OVModelManager: SHA-256 mismatch on attempt %d/%d for '%s'. Expected '%s', got '%s'.",
                  attempt, DownloadMaxAttempts, fullFilePath.GetFullPath(), normalizedExpected.c_str(), normalizedActual.c_str());

               if (wx_file->IsOpened()) {
                  wx_file->Close();
               }
               if (wxFileExists(tempFilePath)) {
                  wxRemoveFile(tempFilePath);
               }

               lastFailure = OVModelManager::InstallResult::Failure(
                  "Downloaded model file failed checksum verification.",
                  BuildInstallDetails(
                     effect,
                     model_info->model_name,
                     "Checksum Verification",
                     "The downloaded file checksum did not match the expected SHA-256 value.",
                     fullFilePath.GetFullPath().ToStdString(),
                     "Attempt: " + std::to_string(attempt) + "/" + std::to_string(DownloadMaxAttempts)
                     + "\nExpected SHA-256: " + normalizedExpected + "\nActual SHA-256: " + normalizedActual + "\nSource URL: " + url));

               if (attempt < DownloadMaxAttempts) {
                  std::this_thread::sleep_for(DownloadRetryDelay);
                  continue;
               }

               return lastFailure;
            }
         }

         if (wx_file->IsOpened()) {
            wx_file->Close();
         }

         if (wxFileExists(oldFilePath)) {
            wxRemoveFile(oldFilePath);
         }

         if (wxFileExists(fullFilePath.GetFullPath())) {
            if (!wxRenameFile(fullFilePath.GetFullPath(), oldFilePath)) {
               if (wxFileExists(tempFilePath)) {
                  wxRemoveFile(tempFilePath);
               }

               return OVModelManager::InstallResult::Failure(
                  "Could not replace an existing model file.",
                  BuildInstallDetails(
                     effect,
                     model_info->model_name,
                     "Install Finalization",
                     "Failed to move the existing model file out of the way before finalizing the download.",
                     fullFilePath.GetFullPath().ToStdString(),
                     "Backup path: " + oldFilePath.ToStdString()));
            }
         }

         if (!wxRenameFile(tempFilePath, fullFilePath.GetFullPath())) {
            if (wxFileExists(oldFilePath)) {
               wxRenameFile(oldFilePath, fullFilePath.GetFullPath());
            }

            return OVModelManager::InstallResult::Failure(
               "Could not finalize the downloaded model file.",
               BuildInstallDetails(
                  effect,
                  model_info->model_name,
                  "Install Finalization",
                  "Failed to move the completed temp file into its final location.",
                  fullFilePath.GetFullPath().ToStdString(),
                  "Temp path: " + tempFilePath.ToStdString()));
         }

         if (wxFileExists(oldFilePath)) {
            wxRemoveFile(oldFilePath);
         }

         wxLogInfo("OVModelManager: SHA-256 for '%s' downloaded from '%s' is %s.",
            fullFilePath.GetFullPath(),
            url.c_str(),
            fileHash.c_str());

         fileDownloaded = true;
         break;
      }

      if (!fileDownloaded) {
         return lastFailure;
      }
   }

   return OVModelManager::InstallResult::Success();
#else
   return OVModelManager::InstallResult::Failure(
      "Model downloads are not available in this build.",
      BuildInstallDetails(effect, model_info ? model_info->model_name : std::string{}, "Download", "download_model_files called, but this build has no networking support."));
#endif
}

OVModelManager::InstallResult OVModelManager::install_model(std::string effect, std::string model_id, ProgressCallback callback)
{
   try {
      auto it = mModelCollection.find(effect);
      if (it == mModelCollection.end()) {
         wxLogError("OVModelManager: model collection for effect '%s' not found.", effect.c_str());
         return InstallResult::Failure(
            "Model install failed before download started.",
            BuildInstallDetails(effect, model_id, "Lookup", "No model collection was found for the requested effect."));
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
         wxLogError("OVModelManager: model info for model_id '%s' not found.", model_id.c_str());
         return InstallResult::Failure(
            "Model install failed before download started.",
            BuildInstallDetails(effect, model_id, "Lookup", "No model info entry was found for the requested model."));
      }

      size_t total_download_size = 0;
      auto sizeResult = install_model_size(model_info, total_download_size);
      if (!sizeResult)
      {
         wxLogError("OVModelManager: install_model_size failed for model '%s'.", model_id.c_str());
         return WrapInstallFailure(
            effect,
            model_id,
            "Size Check",
            "Could not determine how much data must be downloaded for this model.",
            sizeResult);
      }

      if (mSearchPaths.empty()) {
         return InstallResult::Failure(
            "Model install failed before download started.",
            BuildInstallDetails(effect, model_id, "Path Setup", "No model installation directory is configured."));
      }

      auto& base_openvino_models_path = mSearchPaths[0];

      // re-check the dependencies, but force it to use the 'base' installation folder that we will install to.
      for (auto& d : model_info->dependencies) {
         _check_installed_model_impl(d, base_openvino_models_path);
      }

      // add the total size of the dependencies.
      for (auto& d : model_info->dependencies) {
         if (!d->installed) {
            size_t dependencies_size = 0;
            auto dependencySizeResult = install_model_size(d, dependencies_size);
            if (!dependencySizeResult)
            {
               wxLogError("OVModelManager: install_model_size failed for dependency '%s'.", d->model_name.c_str());
               return WrapInstallFailure(
                  effect,
                  model_id,
                  "Dependency Size Check",
                  "Could not determine how much data must be downloaded for a dependency.",
                  dependencySizeResult,
                  d->model_name);
            }

            total_download_size += dependencies_size;
         }
      }

      size_t bytes_downloaded_so_far = 0;

      if (!model_info->dependencies.empty()) {
         for (auto& d : model_info->dependencies) {
            if (!d->installed) {
               auto dependencyDownloadResult = download_model_files(effect, d, base_openvino_models_path, total_download_size, bytes_downloaded_so_far, callback);
               if (!dependencyDownloadResult) {
                  return WrapInstallFailure(
                     effect,
                     model_id,
                     "Dependency Download",
                     "A required dependency failed to download.",
                     dependencyDownloadResult,
                     d->model_name);
               }

               _check_installed_model_impl(d, base_openvino_models_path);
               if (!d->installed) {
                  return InstallResult::Failure(
                     "A required dependency did not verify after download.",
                     BuildInstallDetails(effect, model_id, "Dependency Verification", "Downloaded dependency files were not found during post-download verification.", d->model_name));
               }
            }
         }
      }

      auto downloadResult = download_model_files(effect, model_info, base_openvino_models_path, total_download_size, bytes_downloaded_so_far, callback);
      if (!downloadResult) {
         return downloadResult;
      }

      //re-run file check for this model.
      _check_installed_model_impl(model_info, base_openvino_models_path);

      if (model_info->installed) {
         auto callback_it = mInstallCallbacks.find(effect);
         if (callback_it != mInstallCallbacks.end())
         {
            callback_it->second(model_info->model_name);
         }

         return InstallResult::Success();
      }

      return InstallResult::Failure(
         "Model files did not verify after download.",
         BuildInstallDetails(effect, model_id, "Verification", "Download completed, but the expected installed files were not found in the target directory.", base_openvino_models_path.ToStdString()));
   }
   catch (const std::exception& error) {
      wxLogError("OVModelManager: exception while installing model '%s'. Exception: %s", model_id.c_str(), error.what());
      return InstallResult::Failure(
         "Unexpected exception while installing the model.",
         BuildInstallDetails(effect, model_id, "Exception", "install_model caught an unexpected exception.", {}, error.what()));
   }
}

void OVModelManager::register_installed_callback(const std::string& effect, InstalledCallback callback)
{
   // For the 2nd+ register call, don't insert it.
   if (mInstallCallbacks.count(effect) == 0) {
      mInstallCallbacks.insert({ effect, callback });
   }
}
