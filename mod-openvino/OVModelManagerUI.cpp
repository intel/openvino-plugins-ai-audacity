#include "OVModelManagerUI.h"
#include <thread>


ModelEntryPanel::ModelEntryPanel(wxWindow* parent, const std::string peffect, std::shared_ptr<OVModelManager::ModelInfo> minfo, ModelManagerDialog* mgr)
   : wxPanel(parent), effect(peffect), model(minfo), manager(mgr)
{
   SetMinSize(wxSize(550, 40));

   wxFlexGridSizer* sizer = new wxFlexGridSizer(4, 5, 5);
   sizer->AddGrowableCol(0, 1);

   wxStaticText* nameText = new wxStaticText(this, wxID_ANY, wxString(model->model_name));
   sizer->Add(nameText, 1, wxEXPAND | wxALIGN_CENTER_VERTICAL);

   wxButton* infoBtn = new wxButton(this, wxID_ANY, "Info");
   sizer->Add(infoBtn, 0, wxALIGN_CENTER_VERTICAL);

   installButton = new wxButton(this, wxID_ANY, model->installed ? "Installed" : "Install");
   installButton->Enable(!model->installed);
   if (model->baseUrl.empty()) {
      if (!model->installed) {
         installButton->SetLabelText("Not Installed");
      }
      installButton->Enable(false);
   }
   sizer->Add(installButton, 0, wxALIGN_CENTER_VERTICAL);

   SetSizerAndFit(sizer);

   infoBtn->Bind(wxEVT_BUTTON, &ModelEntryPanel::OnInfo, this);
   installButton->Bind(wxEVT_BUTTON, &ModelEntryPanel::OnInstall, this);
}

void ModelEntryPanel::OnInfo(wxCommandEvent&) {
   wxMessageBox(wxString(model->info), "Model Info", wxOK | wxICON_INFORMATION, this);
}

void ModelEntryPanel::OnInstall(wxCommandEvent&) {
   installButton->Disable();
   manager->QueueInstall(this);
}

void ModelEntryPanel::UpdateStatus() {
   installButton->SetLabelText(model->installed ? "Installed" : "Install");
   installButton->Enable(!model->installed);
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
InstallQueueEntryPanel::InstallQueueEntryPanel(wxWindow* parent, ModelEntryPanel* source)
   : wxPanel(parent), modelPanel(source)
{
   wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);

   label = new wxStaticText(this, wxID_ANY, source->GetModel()->model_name);
   gauge = new wxGauge(this, wxID_ANY, 100, wxDefaultPosition, wxSize(-1, 16));
   gauge->Hide();

   // Add label above
   sizer->Add(label, 1, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 5);
   // Then add the gauge
   sizer->Add(gauge, 1, wxEXPAND | wxALL, 5);

   SetSizerAndFit(sizer);
}

void InstallQueueEntryPanel::SetAsInstalling() {
   label->SetLabel(modelPanel->GetModel()->model_name + " (Installing...)");
   gauge->SetValue(0);
   gauge->Show();

   Layout();                        // Update this panel
   if (GetParent()) GetParent()->Layout();  // Update queue sizer
}

void InstallQueueEntryPanel::SetAsQueued() {
   label->SetLabel(modelPanel->GetModel()->model_name + " (Queued)");
   gauge->Hide();
   Layout();
}

void InstallQueueEntryPanel::UpdateProgress(int percent) {
   gauge->SetValue(percent);
}

ModelEntryPanel* InstallQueueEntryPanel::GetSourcePanel() const {
   return modelPanel;
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

   SetMinSize(wxSize(600, 400));
   wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);

   scrollPanel = new wxScrolledWindow(this, wxID_ANY, wxDefaultPosition, wxSize(-1, 400), wxVSCROLL);
   scrollPanel->SetScrollRate(5, 5);
   scrollPanel->SetMinSize(wxSize(550, 400));
   modelSizer = new wxBoxSizer(wxVERTICAL);
   scrollPanel->SetSizer(modelSizer);

   std::vector < std::string > allSections = {
      OVModelManager::MusicGenName(),
      OVModelManager::MusicSepName(),
      OVModelManager::NoiseSuppressName(),
      OVModelManager::SuperResName(),
      OVModelManager::WhisperName() };

   for (const auto& s : allSections) {
      auto collection = OVModelManager::instance().GetModelCollection(s);
      if (collection->models.empty()) {
         std::cout << "Empty collection for section=" << s << std::endl;
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
         auto* panel = new ModelEntryPanel(scrollPanel, s, m, this);
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

void ModelManagerDialog::QueueInstall(ModelEntryPanel* panel) {
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

void ModelManagerDialog::BeginInstallFor(ModelEntryPanel* panel) {
   std::thread([this, panel]() {

      auto effect = panel->GetEffect();
      auto model_name = panel->GetModel()->model_name;

      OVModelManager::ProgressCallback callback =
         [this](float perc_complete) {
         wxTheApp->CallAfter([=]() {
            if (!queuePanels.empty())
               queuePanels.front()->UpdateProgress(static_cast<int>(perc_complete * 100));
            });
         };

      OVModelManager::instance().install_model(effect, model_name, callback);

      wxTheApp->CallAfter([=]() {
         activeInstall->SetInstalled();

         auto* completedPanel = queuePanels.front();
         queueSizer->Detach(completedPanel);
         completedPanel->Destroy();
         queuePanels.erase(queuePanels.begin());

         activeInstall = nullptr;
         StartNextInstall();  // recursively process queue
         });
      }).detach();
}

void ModelManagerDialog::StartNextInstall() {
   if (installQueue.empty())
      return;

   activeInstall = installQueue.front();
   installQueue.pop();

   auto* queueEntry = queuePanels.front();
   queueEntry->SetAsInstalling();
   BeginInstallFor(queueEntry->GetSourcePanel());

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
