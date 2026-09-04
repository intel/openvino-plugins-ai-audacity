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
#include <wx/textfile.h>

#include <wx/file.h>
#include <wx/log.h>

namespace {

constexpr size_t DownloadBufferSize = 64 * 1024;
constexpr int DownloadMaxAttempts = 3;
constexpr auto DownloadRetryDelay = std::chrono::milliseconds(250);
constexpr char ModelRevisionStampFileName[] = ".ov_model_revision";

std::string BuildInstallDetails(
   const std::string& effect,
   const std::string& modelName,
   const std::string& stage,
   const std::string& message,
   const std::string& resource = {},
   const std::string& extra = {});

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

wxString BuildRevisionStampPath(const FilePath& searchPathBase, const std::string& relativePath)
{
   wxFileName stampPath(searchPathBase + "/" + relativePath + "/" + ModelRevisionStampFileName);
   stampPath.Normalize();
   return stampPath.GetFullPath();
}

bool ReadRevisionStamp(const wxString& stampPath, std::string& revision)
{
   revision.clear();

   if (!wxFileExists(stampPath)) {
      return false;
   }

   wxTextFile stampFile;
   if (!stampFile.Open(stampPath)) {
      return false;
   }

   for (size_t i = 0; i < stampFile.GetLineCount(); ++i) {
      const wxString line = stampFile.GetLine(i);
      if (line.StartsWith("revision=")) {
         revision = line.Mid(9).ToStdString();
         break;
      }
   }

   stampFile.Close();
   return !revision.empty();
}

OVModelManager::InstallResult WriteRevisionStamp(
   const std::string& effect,
   const std::shared_ptr<OVModelManager::ModelInfo>& model_info,
   const FilePath& searchPathBase)
{
   if (!model_info) {
      return OVModelManager::InstallResult::Failure(
         "Model revision stamp write failed.",
         BuildInstallDetails(effect, {}, "Revision Stamp", "write stamp called with null model."));
   }

   const auto revision = model_info->revision;
   if (revision.empty()) {
      return OVModelManager::InstallResult::Success();
   }

   const auto stampPath = BuildRevisionStampPath(searchPathBase, model_info->relative_path);
   if (wxFileExists(stampPath) && !wxRemoveFile(stampPath)) {
      return OVModelManager::InstallResult::Failure(
         "Model revision stamp write failed.",
         BuildInstallDetails(effect, model_info->model_name, "Revision Stamp",
            "Could not remove existing model revision stamp before rewrite.",
            stampPath.ToStdString()));
   }

   wxTextFile stampFile;
   if (!stampFile.Create(stampPath)) {
      return OVModelManager::InstallResult::Failure(
         "Model revision stamp write failed.",
         BuildInstallDetails(effect, model_info->model_name, "Revision Stamp",
            "Could not create model revision stamp file.",
            stampPath.ToStdString()));
   }

   stampFile.AddLine("revision=" + wxString::FromUTF8(revision.c_str()));
   if (!stampFile.Write()) {
      stampFile.Close();
      return OVModelManager::InstallResult::Failure(
         "Model revision stamp write failed.",
         BuildInstallDetails(effect, model_info->model_name, "Revision Stamp",
            "Could not write model revision stamp file.",
            stampPath.ToStdString()));
   }

   stampFile.Close();
   return OVModelManager::InstallResult::Success();
}

std::string BuildInstallDetails(
   const std::string& effect,
   const std::string& modelName,
   const std::string& stage,
   const std::string& message,
   const std::string& resource,
   const std::string& extra)
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
   model_info->update_available = false;
   model_info->installation_path.clear();

   if (!model_info->dependencies.empty())
   {
      for (auto& d : model_info->dependencies) {
         _check_installed_model_impl(d, search_path_base);

         if (!d->installed)
         {
            if (d->update_available) {
               model_info->update_available = true;
            }
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

      const auto expectedRevision = model_info->revision;
      if (!expectedRevision.empty()) {
         std::string installedRevision;
         const auto stampPath = BuildRevisionStampPath(search_path_base, model_info->relative_path);
         if (!ReadRevisionStamp(stampPath, installedRevision)) {
            model_info->update_available = true;
            wxLogInfo("OVModelManager: model '%s' is present but has no readable revision stamp; update/reinstall required.",
               model_info->model_name.c_str());
            return;
         }

         if (installedRevision != expectedRevision) {
            model_info->update_available = true;
            wxLogInfo("OVModelManager: model '%s' revision stamp mismatch (installed='%s', expected='%s'); update/reinstall required.",
               model_info->model_name.c_str(),
               installedRevision.c_str(),
               expectedRevision.c_str());
            return;
         }
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
   if (!model_info) {
      wxLogError("OVModelManager::install_model_size called on null model_info.");
      return InstallResult::Failure(
         "Model size calculation failed.",
         BuildInstallDetails({}, {}, "Size Check", "install_model_size received a null model pointer."));
   }

   for (const auto& file : model_info->files) {
      if (file.expected_size == 0) {
         return InstallResult::Failure(
            "Model size metadata is missing.",
            BuildInstallDetails({}, model_info->model_name, "Size Check",
               "Model manifest lock data did not provide expected_size for one or more files.",
               file.name,
               "All files must include expected_size before download starts."));
      }

      total_size += static_cast<size_t>(file.expected_size);
   }

   return InstallResult::Success();
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
         auto fileBytesReceived = std::make_shared<std::uint64_t>(0);

         response->setOnDataReceivedCallback(
            [response, wx_file, downloadBuffer, fileHasher, fileBytesReceived, tempFilePath, &bError, &bytes_downloaded_so_far, callback, &total_download_size, &file_error_summary, &file_error_details, &effect, model_info, url, fullFilePath, file, attempt](audacity::network_manager::IResponse*)
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
                     *fileBytesReceived += static_cast<std::uint64_t>(bytesRead);

                     if (file.expected_size > 0 && *fileBytesReceived > file.expected_size)
                     {
                        wxLogError("OVModelManager: download exceeded expected file size on attempt %d/%d for '%s'. Expected %llu bytes, received %llu bytes.",
                           attempt,
                           DownloadMaxAttempts,
                           fullFilePath.GetFullPath(),
                           static_cast<unsigned long long>(file.expected_size),
                           static_cast<unsigned long long>(*fileBytesReceived));
                        file_error_summary = "Downloaded model file exceeded expected size.";
                        file_error_details = BuildInstallDetails(
                           effect,
                           model_info->model_name,
                           "Size Verification",
                           "The streamed download exceeded the expected file size from model metadata before completion.",
                           fullFilePath.GetFullPath().ToStdString(),
                           "Attempt: " + std::to_string(attempt) + "/" + std::to_string(DownloadMaxAttempts)
                           + "\nExpected bytes: " + std::to_string(file.expected_size)
                           + "\nBytes received so far: " + std::to_string(*fileBytesReceived)
                           + "\nSource URL: " + url);
                        bError = true;
                        response->Cancel();
                        return;
                     }

                     if (total_download_size > 0 && bytes_downloaded_so_far > total_download_size)
                     {
                        wxLogError("OVModelManager: aggregate download exceeded planned total on attempt %d/%d for '%s'. Planned %llu bytes, downloaded %llu bytes so far.",
                           attempt,
                           DownloadMaxAttempts,
                           fullFilePath.GetFullPath(),
                           static_cast<unsigned long long>(total_download_size),
                           static_cast<unsigned long long>(bytes_downloaded_so_far));
                        file_error_summary = "Downloaded data exceeded planned total size.";
                        file_error_details = BuildInstallDetails(
                           effect,
                           model_info->model_name,
                           "Size Verification",
                           "Accumulated downloaded bytes exceeded the planned total size before completion.",
                           fullFilePath.GetFullPath().ToStdString(),
                           "Attempt: " + std::to_string(attempt) + "/" + std::to_string(DownloadMaxAttempts)
                           + "\nPlanned total bytes: " + std::to_string(total_download_size)
                           + "\nDownloaded bytes so far: " + std::to_string(bytes_downloaded_so_far)
                           + "\nSource URL: " + url);
                        bError = true;
                        response->Cancel();
                        return;
                     }

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

         if (file.expected_size > 0 && *fileBytesReceived != file.expected_size) {
            wxLogWarning("OVModelManager: size mismatch on attempt %d/%d for '%s'. Expected %llu bytes, got %llu bytes.",
               attempt,
               DownloadMaxAttempts,
               fullFilePath.GetFullPath(),
               static_cast<unsigned long long>(file.expected_size),
               static_cast<unsigned long long>(*fileBytesReceived));

            if (wx_file->IsOpened()) {
               wx_file->Close();
            }
            if (wxFileExists(tempFilePath)) {
               wxRemoveFile(tempFilePath);
            }

            lastFailure = OVModelManager::InstallResult::Failure(
               "Downloaded model file size did not match expected metadata.",
               BuildInstallDetails(
                  effect,
                  model_info->model_name,
                  "Size Verification",
                  "The downloaded file size did not match the expected value from the model manifest lock data.",
                  fullFilePath.GetFullPath().ToStdString(),
                  "Attempt: " + std::to_string(attempt) + "/" + std::to_string(DownloadMaxAttempts)
                  + "\nExpected bytes: " + std::to_string(file.expected_size)
                  + "\nActual bytes: " + std::to_string(*fileBytesReceived)
                  + "\nSource URL: " + url));

            if (attempt < DownloadMaxAttempts) {
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

               auto dependencyStampResult = WriteRevisionStamp(effect, d, base_openvino_models_path);
               if (!dependencyStampResult) {
                  return dependencyStampResult;
               }

               _check_installed_model_impl(d, base_openvino_models_path);
               if (!d->installed) {
                  return InstallResult::Failure(
                     "A required dependency revision did not match this plugin build.",
                     BuildInstallDetails(effect, model_id, "Dependency Revision Verification", "Downloaded dependency files are present, but revision stamp verification failed.", d->model_name));
               }
            }
         }
      }

      auto downloadResult = download_model_files(effect, model_info, base_openvino_models_path, total_download_size, bytes_downloaded_so_far, callback);
      if (!downloadResult) {
         return downloadResult;
      }

      auto stampResult = WriteRevisionStamp(effect, model_info, base_openvino_models_path);
      if (!stampResult) {
         return stampResult;
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
