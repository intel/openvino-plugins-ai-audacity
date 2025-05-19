#include "OVModelManagerUI.h"
#include <thread>

ModelEntryPanel::ModelEntryPanel(wxWindow* parent, const ModelData& data, ModelManagerDialog* mgr)
   : wxPanel(parent), model(data), manager(mgr)
{
   SetMinSize(wxSize(550, 40));

   wxFlexGridSizer* sizer = new wxFlexGridSizer(4, 5, 5);
   sizer->AddGrowableCol(0, 1);

   wxStaticText* nameText = new wxStaticText(this, wxID_ANY, model.name);
   sizer->Add(nameText, 1, wxEXPAND | wxALIGN_CENTER_VERTICAL);

   wxButton* infoBtn = new wxButton(this, wxID_ANY, "Info");
   sizer->Add(infoBtn, 0, wxALIGN_CENTER_VERTICAL);

   statusText = new wxStaticText(this, wxID_ANY, model.installed ? "Installed" : "Not Installed");
   sizer->Add(statusText, 0, wxALIGN_CENTER_VERTICAL);

   installButton = new wxButton(this, wxID_ANY, "Install");
   installButton->Enable(!model.installed);
   sizer->Add(installButton, 0, wxALIGN_CENTER_VERTICAL);

   SetSizerAndFit(sizer);

   infoBtn->Bind(wxEVT_BUTTON, &ModelEntryPanel::OnInfo, this);
   installButton->Bind(wxEVT_BUTTON, &ModelEntryPanel::OnInstall, this);
}

void ModelEntryPanel::OnInfo(wxCommandEvent&) {
   wxMessageBox(model.description, "Model Info", wxOK | wxICON_INFORMATION, this);
}

void ModelEntryPanel::OnInstall(wxCommandEvent&) {
   installButton->Disable();
   manager->QueueInstall(this);
}

void ModelEntryPanel::UpdateStatus() {
   statusText->SetLabel(model.installed ? "Installed" : "Not Installed");
   installButton->Enable(!model.installed);
}

void ModelEntryPanel::SetQueued() {
   statusText->SetLabel("Queued");
   installButton->Disable();
}

void ModelEntryPanel::SetInstalling() {
   statusText->SetLabel("Installing...");
   installButton->Disable();
}

void ModelEntryPanel::SetInstalled() {
   model.installed = true;
   UpdateStatus();
}
InstallQueueEntryPanel::InstallQueueEntryPanel(wxWindow* parent, ModelEntryPanel* source)
   : wxPanel(parent), modelPanel(source)
{
   wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);

   label = new wxStaticText(this, wxID_ANY, source->GetModel().name);
   gauge = new wxGauge(this, wxID_ANY, 100, wxDefaultPosition, wxSize(-1, 16));
   gauge->Hide();

   // Add label above
   sizer->Add(label, 1, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 5);
   // Then add the gauge
   sizer->Add(gauge, 1, wxEXPAND | wxALL, 5);

   SetSizerAndFit(sizer);
}

void InstallQueueEntryPanel::SetAsInstalling() {
   label->SetLabel(modelPanel->GetModel().name + " (Installing...)");
   gauge->SetValue(0);
   gauge->Show();

   Layout();                        // Update this panel
   if (GetParent()) GetParent()->Layout();  // Update queue sizer
}

void InstallQueueEntryPanel::SetAsQueued() {
   label->SetLabel(modelPanel->GetModel().name + " (Queued)");
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
   : wxDialog(parent, wxID_ANY, "Model Manager", wxDefaultPosition, wxSize(600, 600)),
   installTimer(this)
{
   SetMinSize(wxSize(600, 400));
   wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);

   scrollPanel = new wxScrolledWindow(this, wxID_ANY, wxDefaultPosition, wxSize(-1, 400), wxVSCROLL);
   scrollPanel->SetScrollRate(5, 5);
   scrollPanel->SetMinSize(wxSize(550, 400));
   modelSizer = new wxBoxSizer(wxVERTICAL);
   scrollPanel->SetSizer(modelSizer);

   std::vector<ModelData> models = {
       {"whisper_tiny", "Whisper Tiny", "A tiny ASR model.", "Whisper", false, 20 * 1024 * 1024},
       {"whisper_base", "Whisper Base", "A base ASR model.", "Whisper", false, 20 * 1024 * 1024},
       {"noise_v1", "NoiseNet v1", "Suppresses background noise.", "Noise Suppression", true, 15 * 1024 * 1024}
   };

   wxString currentSection;
   wxStaticBoxSizer* currentSectionBox = nullptr;
   wxBoxSizer* currentSectionInner = nullptr;

   for (const auto& m : models) {
      if (m.section != currentSection) {
         currentSection = m.section;

         // Create a new section box
         currentSectionInner = new wxBoxSizer(wxVERTICAL);
         currentSectionBox = new wxStaticBoxSizer(wxVERTICAL, scrollPanel, currentSection);
         currentSectionBox->Add(currentSectionInner, 0, wxEXPAND | wxALL, 5);

         modelSizer->Add(currentSectionBox, 0, wxEXPAND | wxALL, 5);
      }

      auto* panel = new ModelEntryPanel(scrollPanel, m, this);
      currentSectionInner->Add(panel, 0, wxEXPAND | wxALL, 2);
      allPanels.push_back(panel);
   }

   scrollPanel->FitInside();
   mainSizer->Add(scrollPanel, 1, wxEXPAND | wxALL, 10);

   wxStaticBoxSizer* queueBox = new wxStaticBoxSizer(wxVERTICAL, this, "Install Queue");
   queueBox->SetMinSize(wxSize(-1, 40));
   queueSizer = new wxBoxSizer(wxVERTICAL);
   queueBox->Add(queueSizer, 1, wxEXPAND | wxALL, 5);
   mainSizer->Add(queueBox, 1, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);

   SetSizerAndFit(mainSizer);
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
      for (int i = 0; i <= 100; i += 10) {
         std::this_thread::sleep_for(std::chrono::milliseconds(500));
         wxTheApp->CallAfter([=]() {
            if (!queuePanels.empty())
               queuePanels.front()->UpdateProgress(i);
            });
      }

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
