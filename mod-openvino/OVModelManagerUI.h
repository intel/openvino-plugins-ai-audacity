#pragma once

#include <wx/wx.h>
#include <wx/listctrl.h>
#include <wx/string.h>
#include <wx/scrolwin.h>
#include <wx/timer.h>
#include <queue>

struct ModelData {
   wxString id;
   wxString name;
   wxString description;
   wxString section;
   bool installed = false;
   size_t sizeBytes = 0;
};

void ShowModelManagerDialog();

class ModelEntryPanel;
class InstallQueueEntryPanel;
class ModelManagerDialog : public wxDialog {
public:
   static void ShowDialog();
   void QueueInstall(ModelEntryPanel* panel);

private:
   ModelManagerDialog(wxWindow* parent);
   static ModelManagerDialog* instance;

   void StartNextInstall();
   void BeginInstallFor(ModelEntryPanel* panel);

   wxScrolledWindow* scrollPanel;
   wxBoxSizer* modelSizer;
   wxBoxSizer* queueSizer;
   wxTimer installTimer;

   std::vector<ModelEntryPanel*> allPanels;
   std::vector<InstallQueueEntryPanel*> queuePanels;
   std::queue<ModelEntryPanel*> installQueue;
   ModelEntryPanel* activeInstall = nullptr;
   int installProgress = 0;

   wxDECLARE_EVENT_TABLE();
};

class ModelEntryPanel : public wxPanel {
public:
   ModelEntryPanel(wxWindow* parent, const ModelData& data, ModelManagerDialog* manager);

   const ModelData& GetModel() const { return model; }

   void UpdateStatus();
   void SetQueued();
   void SetInstalling();
   void SetInstalled();

private:
   void OnInfo(wxCommandEvent& event);
   void OnInstall(wxCommandEvent& event);

   ModelData model;
   ModelManagerDialog* manager;

   wxStaticText* statusText;
   wxButton* installButton;
};

class InstallQueueEntryPanel : public wxPanel {
public:
   InstallQueueEntryPanel(wxWindow* parent, ModelEntryPanel* sourcePanel);

   void SetAsInstalling();
   void SetAsQueued();
   void UpdateProgress(int percent);
   ModelEntryPanel* GetSourcePanel() const;

private:
   ModelEntryPanel* modelPanel;
   wxStaticText* label;
   wxGauge* gauge;
};

