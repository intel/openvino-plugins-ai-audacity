#include "OVModelManagerUI.h"
#include "OpenVINOPluginPrefs.h"
#include <algorithm>
#include <thread>
#include <wx/clipbrd.h>
#include <wx/dataobj.h>
#include <wx/html/htmlwin.h>
#include <wx/log.h>
#include <wx/regex.h>
#include <wx/settings.h>
#include <wx/textctrl.h>

class ModelCardDialog : public wxDialog {
public:
   ModelCardDialog(wxWindow* parent, const wxString& title, const wxString& htmlContent)
      : wxDialog(parent, wxID_ANY, title, wxDefaultPosition, wxSize(600, 500),
         wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER)
   {
      wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

      wxHtmlWindow* htmlWindow = new wxHtmlWindow(this, wxID_ANY,
         wxDefaultPosition,
         wxDefaultSize,
         wxHW_SCROLLBAR_AUTO);
      htmlWindow->SetPage(htmlContent);

      // Make <a href="..."> links clickable
      htmlWindow->Bind(wxEVT_HTML_LINK_CLICKED, [=](wxHtmlLinkEvent& event) {
         wxLaunchDefaultBrowser(event.GetLinkInfo().GetHref());
         });

      sizer->Add(htmlWindow, 1, wxEXPAND | wxALL, 10);

      wxButton* closeBtn = new wxButton(this, wxID_OK, "Close");
      sizer->Add(closeBtn, 0, wxALIGN_RIGHT | wxALL, 10);

      SetSizer(sizer);
      Layout();
      CentreOnParent();
   }
};

class InstallFailureDialog : public wxDialog {
public:
   InstallFailureDialog(wxWindow* parent, const wxString& title, const wxString& details)
      : wxDialog(parent, wxID_ANY, title, wxDefaultPosition, wxSize(700, 420),
         wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER)
   {
      wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);

      auto* detailsText = new wxTextCtrl(
         this,
         wxID_ANY,
         details,
         wxDefaultPosition,
         wxDefaultSize,
         wxTE_MULTILINE | wxTE_READONLY | wxTE_DONTWRAP);
      mainSizer->Add(detailsText, 1, wxEXPAND | wxALL, 10);

      wxBoxSizer* buttonSizer = new wxBoxSizer(wxHORIZONTAL);
      auto* copyButton = new wxButton(this, wxID_ANY, "Copy Details");
      auto* closeButton = new wxButton(this, wxID_OK, "Close");

      copyButton->Bind(wxEVT_BUTTON, [details, this](wxCommandEvent&) {
         if (wxTheClipboard && wxTheClipboard->Open()) {
            wxTheClipboard->SetData(new wxTextDataObject(details));
            wxTheClipboard->Close();
         }
      });

      buttonSizer->Add(copyButton, 0, wxRIGHT, 8);
      buttonSizer->Add(closeButton, 0);
      mainSizer->Add(buttonSizer, 0, wxALIGN_RIGHT | wxLEFT | wxRIGHT | wxBOTTOM, 10);

      SetSizer(mainSizer);
      Layout();
      CentreOnParent();
   }
};

ModelEntryPanel::ModelEntryPanel(wxWindow* parent, const std::string peffect, std::shared_ptr<OVModelManager::ModelInfo> minfo, ModelManagerDialog* mgr, bool restartReq)
   : wxPanel(parent), effect(peffect), model(minfo), manager(mgr), restartRequired(restartReq)
{
   SetMinSize(wxSize(550, 40));

   wxFlexGridSizer* sizer = new wxFlexGridSizer(4, 5, 5);
   sizer->AddGrowableCol(0, 1);

   wxStaticText* nameText = new wxStaticText(this, wxID_ANY, wxString(model->model_name));
   sizer->Add(nameText, 1, wxEXPAND | wxALIGN_CENTER_VERTICAL);

   wxButton* infoBtn = new wxButton(this, wxID_ANY, "Model Card");
   sizer->Add(infoBtn, 0, wxALIGN_CENTER_VERTICAL);

   installButton = new wxButton(this, wxID_ANY, model->installed ? "Installed" : "Install");
   installButton->Enable(!model->installed && !restartRequired);
   if (model->baseUrl.empty()) {
      if (!model->installed) {
         installButton->SetLabelText("Not Installed");
      }
      installButton->Enable(false);
   }
#ifndef HAS_NETWORKING
   // if we don't have networking support, disable (grey-out) install button.
   installButton->Enable(false);
#endif

   if (restartRequired && !model->installed) {
      installButton->SetLabelText("Restart Required");
      installButton->Enable(false);
   }

   sizer->Add(installButton, 0, wxALIGN_CENTER_VERTICAL);

   SetSizerAndFit(sizer);

   infoBtn->Bind(wxEVT_BUTTON, &ModelEntryPanel::OnInfo, this);
   installButton->Bind(wxEVT_BUTTON, &ModelEntryPanel::OnInstall, this);
}

void ModelEntryPanel::OnInfo(wxCommandEvent&) {
   // Assuming `model` is a member variable or otherwise accessible
   wxString html = model->info;  // model->info is a markdown string

   // Create and show the dialog
   ModelCardDialog* dlg = new ModelCardDialog(this, "Model Card", html);
   dlg->ShowModal();
   dlg->Destroy();
}

void ModelEntryPanel::OnInstall(wxCommandEvent&) {
   installButton->Disable();
   manager->QueueInstall(this);
}

void ModelEntryPanel::UpdateStatus() {
   if (restartRequired && !model->installed) {
      installButton->SetLabelText("Restart Required");
      installButton->Enable(false);
      installButton->SetToolTip({});
      return;
   }

   installButton->SetLabelText(model->installed ? "Installed" : "Install");
   installButton->Enable(!model->installed);
   installButton->SetToolTip({});
}

void ModelEntryPanel::SetQueued() {
   installButton->SetLabelText("Queued");
   installButton->Disable();
}

void ModelEntryPanel::SetInstalling() {
   installButton->SetLabelText("Installing...");
   installButton->Disable();
}

void ModelEntryPanel::SetInstalled() {
   UpdateStatus();
}

void ModelEntryPanel::SetFailed(const wxString& summary) {
   if (restartRequired && !model->installed) {
      UpdateStatus();
      return;
   }

   installButton->SetLabelText("Retry Install");
   installButton->Enable(!model->installed);
   installButton->SetToolTip(summary);
}

InstallQueueEntryPanel::InstallQueueEntryPanel(wxWindow* parent, ModelEntryPanel* source)
   : wxPanel(parent), modelPanel(source)
{
   wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);

   label = new wxStaticText(this, wxID_ANY, source->GetModel()->model_name);
   gauge = new wxGauge(this, wxID_ANY, 100, wxDefaultPosition, wxSize(-1, 16));
   detailsButton = new wxButton(this, wxID_ANY, "Details");
   detailsButton->SetInitialSize(detailsButton->GetBestSize());
   gauge->Hide();
   detailsButton->Disable();

   // Add label above
   sizer->Add(label, 1, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 5);
   // Then add the gauge
   sizer->Add(gauge, 1, wxEXPAND | wxALL, 5);
   sizer->Add(detailsButton, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);

   SetSizerAndFit(sizer);

   detailsButton->Bind(wxEVT_BUTTON, &InstallQueueEntryPanel::OnViewDetails, this);
}

void InstallQueueEntryPanel::SetAsInstalling() {
   label->SetLabel(modelPanel->GetModel()->model_name + " (Installing...)");
   label->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));
   gauge->SetValue(0);
   gauge->Show();
   detailsButton->Disable();
   detailsButton->SetToolTip({});

   Layout();                        // Update this panel
   if (GetParent()) GetParent()->Layout();  // Update queue sizer
}

void InstallQueueEntryPanel::SetAsQueued() {
   label->SetLabel(modelPanel->GetModel()->model_name + " (Queued)");
   label->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));
   gauge->Hide();
   detailsButton->Disable();
   detailsButton->SetToolTip({});
   Layout();
}

void InstallQueueEntryPanel::SetAsFailed(const OVModelManager::InstallResult& result) {
   installResult = result;
   label->SetLabel(modelPanel->GetModel()->model_name + " (Failed)");
   label->SetForegroundColour(wxColour(160, 0, 0));
   gauge->Hide();
   detailsButton->Enable();
   detailsButton->SetToolTip(result.summary);

   Layout();
   if (GetParent()) {
      GetParent()->Layout();
   }
}

void InstallQueueEntryPanel::UpdateProgress(int percent) {
   gauge->SetValue(percent);
}

ModelEntryPanel* InstallQueueEntryPanel::GetSourcePanel() const {
   return modelPanel;
}

void InstallQueueEntryPanel::OnViewDetails(wxCommandEvent&) {
   wxString details = installResult.details.empty() ? installResult.summary : installResult.details;
   InstallFailureDialog dialog(this, "Install Failure Details", details);
   dialog.ShowModal();
}

ModelManagerDialog* ModelManagerDialog::instance = nullptr;

wxBEGIN_EVENT_TABLE(ModelManagerDialog, wxDialog)
wxEND_EVENT_TABLE()

void ModelManagerDialog::ShowDialog() {
   if (!instance) {
      instance = new ModelManagerDialog(wxTheApp->GetTopWindow());
      instance->Show();
   }
   else {
      instance->Show();
   }
}

ModelManagerDialog::ModelManagerDialog(wxWindow* parent)
   : wxDialog(parent, wxID_ANY, "Model Manager", wxDefaultPosition, wxSize(600, 600), wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER),
   installTimer(this)
{
   // trigger constructions of OVModelManager
   {
      OVModelManager::instance();
   }

   const bool restartRequired = OpenVINOPluginSettings::IsModelDirRestartRequiredThisSession();

   SetMinSize(wxSize(600, 400));
   wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);

   scrollPanel = new wxScrolledWindow(this, wxID_ANY, wxDefaultPosition, wxSize(-1, 400), wxVSCROLL);
   scrollPanel->SetScrollRate(5, 5);
   scrollPanel->SetMinSize(wxSize(550, 400));
   modelSizer = new wxBoxSizer(wxVERTICAL);
   scrollPanel->SetSizer(modelSizer);

   if (restartRequired) {
      auto* restartMsg = new wxStaticText(this, wxID_ANY,
         "Please restart Audacity to install models (model directory changed).");
      mainSizer->Add(restartMsg, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
   }

   std::vector < std::string > allSections = {
      OVModelManager::MusicSepName(),
      OVModelManager::MusicRestorationName(),
      OVModelManager::NoiseSuppressName(),
      OVModelManager::ReverbRemovalName(),
      OVModelManager::SuperResName(),
      OVModelManager::TtsName(),
      OVModelManager::WhisperName()
      };

   for (const auto& s : allSections) {
      auto collection = OVModelManager::instance().GetModelCollection(s);
      if (collection->models.empty()) {
         wxLogInfo("OVModelManagerUI: skipping empty model collection section '%s'.", s.c_str());
         continue;
      }

      auto currentSection = wxString(s);
      // Create a new section box
      auto* staticBox = new wxStaticBox(scrollPanel, wxID_ANY, currentSection);
      wxStaticBoxSizer* currentSectionBox = new wxStaticBoxSizer(staticBox, wxVERTICAL);
      // Make label bold
      wxFont font = staticBox->GetFont();
      font.SetWeight(wxFONTWEIGHT_BOLD);
      staticBox->SetFont(font);

      wxBoxSizer* currentSectionInner = new wxBoxSizer(wxVERTICAL);

      currentSectionBox->Add(currentSectionInner, 0, wxEXPAND | wxALL, 5);
      modelSizer->Add(currentSectionBox, 0, wxEXPAND | wxALL, 5);

      for (auto& m : collection->models)
      {
         auto* panel = new ModelEntryPanel(scrollPanel, s, m, this, restartRequired);
         currentSectionInner->Add(panel, 0, wxEXPAND | wxALL, 2);
         allPanels.push_back(panel);
      }
   }

   scrollPanel->FitInside();
   mainSizer->Add(scrollPanel, 7, wxEXPAND | wxALL, 10);

   wxStaticBoxSizer* queueBox = new wxStaticBoxSizer(wxVERTICAL, this, "Install Queue");
   queueBox->SetMinSize(wxSize(-1, 40));
   queueSizer = new wxBoxSizer(wxVERTICAL);
   queueBox->Add(queueSizer, 1, wxEXPAND | wxALL, 5);
   mainSizer->Add(queueBox, 2, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);

   SetSizerAndFit(mainSizer);

   OVModelManager& model_manager = OVModelManager::instance();
}

ModelManagerDialog::~ModelManagerDialog() {
   if (instance == this) {
      instance = nullptr;
   }
}

void ModelManagerDialog::QueueInstall(ModelEntryPanel* panel) {
   if (auto* existingEntry = FindQueueEntry(panel)) {
      RemoveQueueEntry(existingEntry);
   }

   panel->SetQueued();

   auto* entry = new InstallQueueEntryPanel(this, panel);
   entry->SetAsQueued();
   queueSizer->Add(entry, 0, wxEXPAND | wxALL, 2);
   queuePanels.push_back(entry);
   Layout();

   installQueue.push(panel);

   if (!activeInstall)
      StartNextInstall();
}

void ModelManagerDialog::BeginInstallFor(ModelEntryPanel* panel, InstallQueueEntryPanel* queueEntry) {
   if (!panel) {
      wxLogError("BeginInstallFor called with null ModelEntryPanel");
      return;
   }
   const int panelId = panel ? panel->GetId() : wxID_NONE;
   const int queueEntryId = queueEntry ? queueEntry->GetId() : wxID_NONE;
   const auto effect = panel->GetEffect();
   const auto model_name = panel->GetModel()->model_name;

   std::thread([panelId, queueEntryId, effect, model_name]() {

      OVModelManager::ProgressCallback callback =
         [panelId, queueEntryId](float perc_complete) {
         wxTheApp->CallAfter([panelId, queueEntryId, perc_complete]() {
            auto* dialog = ModelManagerDialog::instance;
            if (!dialog || dialog->IsBeingDeleted()) {
               return;
            }

            if (!dialog->activeInstall || dialog->activeInstall->GetId() != panelId) {
               return;
            }

            auto* liveQueueEntry = dialog->FindQueueEntryById(queueEntryId);
            if (!liveQueueEntry || liveQueueEntry->IsBeingDeleted()) {
               return;
            }

            liveQueueEntry->UpdateProgress(static_cast<int>(perc_complete * 100));
            });
         };

      const auto installResult = OVModelManager::instance().install_model(effect, model_name, callback);

      wxTheApp->CallAfter([panelId, queueEntryId, installResult]() {
         auto* dialog = ModelManagerDialog::instance;
         if (!dialog || dialog->IsBeingDeleted()) {
            return;
         }

         auto* livePanel = dialog->FindModelPanelById(panelId);
         const bool panelIsValid = livePanel && !livePanel->IsBeingDeleted();

         auto* liveQueueEntry = dialog->FindQueueEntryById(queueEntryId);
         const bool queueEntryIsValid = liveQueueEntry && !liveQueueEntry->IsBeingDeleted();

         if (installResult) {
            if (panelIsValid) {
               livePanel->SetInstalled();
            }

            if (queueEntryIsValid) {
               dialog->RemoveQueueEntry(liveQueueEntry);
            }
         }
         else {
            if (panelIsValid) {
               livePanel->SetFailed(wxString::FromUTF8(installResult.summary.c_str()));
            }

            if (queueEntryIsValid) {
               liveQueueEntry->SetAsFailed(installResult);
            }
         }

         if (dialog->activeInstall && dialog->activeInstall->GetId() == panelId) {
            dialog->activeInstall = nullptr;
         }

         if (dialog->activeQueueEntry && dialog->activeQueueEntry->GetId() == queueEntryId) {
            dialog->activeQueueEntry = nullptr;
         }

         dialog->StartNextInstall();  // recursively process queue
         });
      }).detach();
}

ModelEntryPanel* ModelManagerDialog::FindModelPanelById(int panelId) const {
   if (panelId == wxID_NONE) {
      return nullptr;
   }

   for (auto* panel : allPanels) {
      if (panel && panel->GetId() == panelId) {
         return panel;
      }
   }

   return nullptr;
}

void ModelManagerDialog::StartNextInstall() {
   if (installQueue.empty())
      return;

   activeInstall = installQueue.front();
   installQueue.pop();

   activeQueueEntry = FindQueueEntry(activeInstall);
   if (activeQueueEntry) {
      activeQueueEntry->SetAsInstalling();
   }

   BeginInstallFor(activeInstall, activeQueueEntry);

   queueSizer->Layout();
   Layout();
}

InstallQueueEntryPanel* ModelManagerDialog::FindQueueEntry(ModelEntryPanel* panel) const {
   for (auto* entry : queuePanels) {
      if (entry && entry->GetSourcePanel() == panel) {
         return entry;
      }
   }

   return nullptr;
}

InstallQueueEntryPanel* ModelManagerDialog::FindQueueEntryById(int entryId) const {
   if (entryId == wxID_NONE) {
      return nullptr;
   }

   for (auto* entry : queuePanels) {
      if (entry && entry->GetId() == entryId) {
         return entry;
      }
   }

   return nullptr;
}

void ModelManagerDialog::RemoveQueueEntry(InstallQueueEntryPanel* entry) {
   if (!entry) {
      return;
   }

   if (activeQueueEntry == entry) {
      activeQueueEntry = nullptr;
   }

   queueSizer->Detach(entry);
   queuePanels.erase(std::remove(queuePanels.begin(), queuePanels.end(), entry), queuePanels.end());
   entry->Destroy();

   queueSizer->Layout();
   Layout();
}

class DeferredModelManagerLauncher : public wxEvtHandler
{
public:
   static void Launch()
   {
      wxIdleEvent* evt = new wxIdleEvent();
      wxTheApp->QueueEvent(evt);
      wxTheApp->Bind(wxEVT_IDLE, &DeferredModelManagerLauncher::OnIdle, new DeferredModelManagerLauncher());
   }

private:
   void OnIdle(wxIdleEvent& event)
   {
      wxTheApp->Unbind(wxEVT_IDLE, &DeferredModelManagerLauncher::OnIdle, this);
      ModelManagerDialog::ShowDialog();
      delete this;
   }
};

void ShowModelManagerDialog()
{
   DeferredModelManagerLauncher::Launch();
}
