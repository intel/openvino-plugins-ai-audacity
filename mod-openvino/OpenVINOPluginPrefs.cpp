// OpenVINOPluginSettings.cpp
// One-stop implementation: prefs helpers + Preferences UI page + registration.

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996) // avoid /WX breaks from Audacity deprecation warnings in headers
#endif

#include "PrefsPanel.h"
#include "ShuttleGui.h"
#include "Prefs.h"
#include "FileNames.h"
#include "MemoryX.h"
#include "Internat.h"

#include <wx/filename.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/checkbox.h>
#include <wx/dirdlg.h>
#include <wx/log.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "OpenVINOPluginPrefs.h"
#include "OVModelManagerUI.h"
#include "AudacityMessageBox.h"

namespace OpenVINOPluginSettings
{
   // Preference keys (stored in audacity.cfg)
   inline constexpr const wchar_t* kPrefModelDir = L"/OpenVINOPlugins/ModelDir";
   inline constexpr const wchar_t* kPrefEnableCache = L"/OpenVINOPlugins/EnableModelCache";
   inline constexpr const wchar_t* kPrefCompiledCache = L"/OpenVINOPlugins/CompiledModelCacheDir";
   static bool gModelDirRestartRequiredThisSession = false;


   static inline std::optional<std::string> get_env_var(const std::string& name) {
      if (const char* value = std::getenv(name.c_str())) {
         return std::string(value); // copy into std::string
      }
      return std::nullopt; // not found
   }

   static wxString DefaultModelsDirPath()
   {
      // if AUDACITY_OPENVINO_MODELS_PATH env is set, return that path.
      auto model_path_from_env = get_env_var("AUDACITY_OPENVINO_MODELS_PATH");
      if (model_path_from_env) {
         return wxFileName(wxString(*model_path_from_env)).GetFullPath();
      }

      // Otherwise, return the default path under Audacity's data directory.
      return wxFileName(FileNames::DataDir(), wxT("openvino-models")).GetFullPath();
   }

   static wxString DefaultCacheDirPath()
   {
      return wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath();
   }

   static wxString EnsureDirPref(const wchar_t* prefKey,
      const wxString& defaultPath,
      bool persistIfMissing)
   {
      wxString value = gPrefs->Read(prefKey, wxString{});
      if (value.empty())
         value = defaultPath;

      const wxString requestedPath = value;
      value = FileNames::MkDir(value);

      if (value.empty() || !wxFileName::DirExists(value)) {
         wxLogWarning("OpenVINO preferences: unable to ensure directory exists for key '%s'. Requested='%s', Result='%s'",
            wxString(prefKey), requestedPath, value);
      }

      if (persistIfMissing) {
         const wxString current = gPrefs->Read(prefKey, wxString{});
         if (current.empty() || current != value)
            gPrefs->Write(prefKey, value);
      }

      return value;
   }


   wxString GetOrCreateModelDir(bool persistIfMissing)
   {
      return EnsureDirPref(kPrefModelDir, DefaultModelsDirPath(), persistIfMissing);
   }

   wxString GetOrCreateCompiledModelCacheDir(bool persistIfMissing)
   {
      return EnsureDirPref(kPrefCompiledCache, DefaultCacheDirPath(), persistIfMissing);
   }

   bool ReadEnableCache(bool defaultValue)
   {
      return gPrefs->ReadBool(kPrefEnableCache, defaultValue);
   }

   bool IsModelDirRestartRequiredThisSession()
   {
      return gModelDirRestartRequiredThisSession;
   }

   void MarkModelDirRestartRequiredThisSession()
   {
      gModelDirRestartRequiredThisSession = true;
   }

} // namespace OpenVINOPluginSettings

// -----------------------------
// Preferences UI page
// -----------------------------
namespace {

   enum
   {
      kModelDirTextID = 5000,
      kModelDirBrowseID,

      kOpenModelManagerButtonID,

      kEnableCacheCheckID,
      kCacheDirTextID,
      kCacheDirBrowseID,
   };

   class OpenVINOPluginsPrefs final : public PrefsPanel
   {
   public:
      OpenVINOPluginsPrefs(wxWindow* parent, wxWindowID winid, AudacityProject* project)
         : PrefsPanel(parent, winid, XO("OpenVINO Plugins"))
      {
         Populate();

         mHostDialog = wxGetTopLevelParent(this);
         if (mHostDialog) {
            mHostDialog->Bind(wxEVT_BUTTON, &OpenVINOPluginsPrefs::OnDialogConfirm, this, wxID_OK);
            mHostDialog->Bind(wxEVT_BUTTON, &OpenVINOPluginsPrefs::OnDialogConfirm, this, wxID_APPLY);
         }
      }

      ~OpenVINOPluginsPrefs() override
      {
         if (mHostDialog) {
            mHostDialog->Unbind(wxEVT_BUTTON, &OpenVINOPluginsPrefs::OnDialogConfirm, this, wxID_OK);
            mHostDialog->Unbind(wxEVT_BUTTON, &OpenVINOPluginsPrefs::OnDialogConfirm, this, wxID_APPLY);
         }
      }

      ComponentInterfaceSymbol GetSymbol() const override
      {
         return ComponentInterfaceSymbol{ wxT("OpenVINOPluginsPrefs") };
      }

      TranslatableString GetDescription() const override
      {
         return XO("Preferences for OpenVINO-based AI plugins");
      }

      void Populate()
      {
         // Ensure defaults exist so other code can just Read() them later
         (void)OpenVINOPluginSettings::GetOrCreateModelDir(true);
         (void)OpenVINOPluginSettings::GetOrCreateCompiledModelCacheDir(true);

         // If the pref was never set before, set it to true (and persist it)
         if (!gPrefs->HasEntry(OpenVINOPluginSettings::kPrefEnableCache)) {
            gPrefs->Write(OpenVINOPluginSettings::kPrefEnableCache, true);
         }
         // Snapshot the value that was active when the prefs page opened.
         // (Read raw, but fall back to the resolved default to avoid empty comparisons.)
         mOriginalModelDir = gPrefs->Read(OpenVINOPluginSettings::kPrefModelDir, wxString{});
         if (mOriginalModelDir.empty())
            mOriginalModelDir = OpenVINOPluginSettings::GetOrCreateModelDir(/*persistIfMissing=*/true);


         ShuttleGui S(this, eIsCreatingFromPrefs);
         PopulateOrExchange(S);

         UpdateCacheControls();
      }

      void PopulateOrExchange(ShuttleGui& S) override
      {
         S.SetBorder(2);
         S.StartScroller();

         S.StartStatic(XO("Models"));
         {
            S.StartMultiColumn(3, wxEXPAND);
            {
               S.SetStretchyCol(1);

               S.Id(kModelDirTextID);
               mModelDirText = S.TieTextBox(
                  XO("Model Install Directory:"),
                  { OpenVINOPluginSettings::kPrefModelDir,
                    OpenVINOPluginSettings::GetOrCreateModelDir(true) },
                  30
               );
               S.Id(kModelDirBrowseID).AddButton(XO("Browse..."));

               S.Id(kOpenModelManagerButtonID).AddButton(XO("Open Model Manager"));
               S.AddSpace(2);
            }
            S.EndMultiColumn();
         }
         S.EndStatic();

         S.StartStatic(XO("Caching"));
         {
            S.StartMultiColumn(3, wxEXPAND);
            {
               S.SetStretchyCol(1);

               S.Id(kEnableCacheCheckID);
               mEnableCacheCheck = S.TieCheckBox(
                  XO("Enable Compiled Model Caching"),
                  OpenVINOPluginSettings::kPrefEnableCache
               );

               S.AddSpace(1);
               S.AddSpace(1);

               S.Id(kCacheDirTextID);
               mCacheDirText = S.TieTextBox(
                  XO("Compiled Model Cache Directory:"),
                  { OpenVINOPluginSettings::kPrefCompiledCache,
                    OpenVINOPluginSettings::GetOrCreateCompiledModelCacheDir(true) },
                  30
               );
               S.Id(kCacheDirBrowseID).AddButton(XO("Browse..."));
            }
            S.EndMultiColumn();
         }
         S.EndStatic();

         S.EndScroller();
      }

      bool Commit() override
      {
         if (!ValidatePendingDirectories(/*reportErrors=*/true))
            return false;

         ShuttleGui S(this, eIsSavingToPrefs);
         PopulateOrExchange(S);

         // Read back the saved value
         wxString newModelDir = gPrefs->Read(OpenVINOPluginSettings::kPrefModelDir, wxString{});

         // Normalize a bit to reduce false positives
         // (Windows paths in wxString are usually case-insensitive; also trim whitespace)
         auto normalize = [](wxString s) {
            s.Trim(true).Trim(false);
#if defined(__WXMSW__)
            s.MakeLower();
#endif
            // Strip trailing separators (optional)
            while (s.EndsWith(wxFILE_SEP_PATH))
               s.RemoveLast();
            return s;
            };

         if (!mOriginalModelDir.empty() && !newModelDir.empty() &&
            normalize(mOriginalModelDir) != normalize(newModelDir))
         {
            OpenVINOPluginSettings::MarkModelDirRestartRequiredThisSession();
            AudacityMessageBox(
               XO("The installed model directory has been updated. Please close & re-open Audacity!"),
               XO("Restart Required"),
               wxOK | wxCENTRE | wxICON_INFORMATION
            );

            // Update snapshot so repeated Apply clicks don't re-pop the dialog
            mOriginalModelDir = newModelDir;
         }


         return true;
      }

   private:
      bool ValidatePendingDirectories(bool reportErrors)
      {
         const wxString pendingModelDir = mModelDirText ? mModelDirText->GetValue() : wxString{};
         const wxString pendingCacheDir = mCacheDirText ? mCacheDirText->GetValue() : wxString{};
         const bool pendingEnableCache = mEnableCacheCheck ? mEnableCacheCheck->GetValue() : true;

         auto validateDirOrReport = [this, reportErrors](const wchar_t* prefKey,
            const wxString& path,
            const TranslatableString& label,
            wxTextCtrl* controlToFocus) -> bool
            {
               if (IsDirectoryWritable(path))
                  return true;

               if (reportErrors) {
                  wxLogWarning("OpenVINO preferences: selected path for key '%s' is not writable: '%s'",
                     wxString(prefKey), path);

                  AudacityMessageBox(
                     XO("The selected %s is not writable (or cannot be created). Please choose a writable directory.")
                     .Format(label),
                     XO("Invalid Directory"),
                     wxOK | wxCENTRE | wxICON_WARNING
                  );

                  if (controlToFocus)
                     controlToFocus->SetFocus();
               }

               return false;
            };

         if (!validateDirOrReport(OpenVINOPluginSettings::kPrefModelDir,
            pendingModelDir,
            XO("model install directory"),
            mModelDirText))
            return false;

         if (pendingEnableCache &&
            !validateDirOrReport(OpenVINOPluginSettings::kPrefCompiledCache,
               pendingCacheDir,
               XO("compiled model cache directory"),
               mCacheDirText))
            return false;

         return true;
      }

      static bool IsDirectoryWritable(wxString path)
      {
         path.Trim(true).Trim(false);
         if (path.empty())
            return false;

         const wxString ensuredPath = FileNames::MkDir(path);
         return !ensuredPath.empty() &&
            wxFileName::DirExists(ensuredPath) &&
            wxFileName::IsDirWritable(ensuredPath);
      }

      void UpdateCacheControls()
      {
         const bool enabled = mEnableCacheCheck && mEnableCacheCheck->GetValue();

         if (mCacheDirText)
            mCacheDirText->Enable(enabled);

         if (auto* btn = wxDynamicCast(FindWindow(kCacheDirBrowseID), wxButton))
            btn->Enable(enabled);
      }

      void OnToggleCache(wxCommandEvent&)
      {
         UpdateCacheControls();
      }

      static void BrowseIntoTextCtrl(wxWindow* parent,
         wxTextCtrl* tc,
         const TranslatableString& title)
      {
         if (!tc)
            return;

         wxDirDialogWrapper dlog(parent, title, tc->GetValue());
         if (dlog.ShowModal() == wxID_CANCEL)
            return;

         const wxString path = dlog.GetPath();
         if (!path.empty())
            tc->SetValue(path);
      }

      void OnBrowseModelDir(wxCommandEvent&)
      {
         BrowseIntoTextCtrl(this, mModelDirText, XO("Choose a model directory"));
      }

      void OnBrowseCacheDir(wxCommandEvent&)
      {
         BrowseIntoTextCtrl(this, mCacheDirText, XO("Choose a compiled model cache directory"));
      }

      void OnOpenModelManager(wxCommandEvent&)
      {
         ShowModelManagerDialog();
      }

      void OnDialogConfirm(wxCommandEvent& event)
      {
         if (!ValidatePendingDirectories(/*reportErrors=*/true))
            return;

         event.Skip();
      }

   private:
      wxTextCtrl* mModelDirText{ nullptr };
      wxCheckBox* mEnableCacheCheck{ nullptr };
      wxTextCtrl* mCacheDirText{ nullptr };
      wxWindow* mHostDialog{ nullptr };

      wxString mOriginalModelDir;

      wxDECLARE_EVENT_TABLE();
   };

   wxBEGIN_EVENT_TABLE(OpenVINOPluginsPrefs, PrefsPanel)
      EVT_BUTTON(kModelDirBrowseID, OpenVINOPluginsPrefs::OnBrowseModelDir)
      EVT_BUTTON(kOpenModelManagerButtonID, OpenVINOPluginsPrefs::OnOpenModelManager)
      EVT_BUTTON(kCacheDirBrowseID, OpenVINOPluginsPrefs::OnBrowseCacheDir)
      EVT_CHECKBOX(kEnableCacheCheckID, OpenVINOPluginsPrefs::OnToggleCache)
      wxEND_EVENT_TABLE()

      // Registration: static initializer invoked when your module DLL is loaded.
      PrefsPanel::Registration sAttachment{
         wxT("OpenVINO Plugins"),
         [](wxWindow* parent, wxWindowID winid, AudacityProject* project) -> PrefsPanel* {
            wxASSERT(parent);
            return safenew OpenVINOPluginsPrefs(parent, winid, project);
         },
         true
   };

} // anonymous namespace
