#pragma once

#include <wx/wx.h>
#include <wx/listctrl.h>
#include <wx/string.h>
#include <wx/scrolwin.h>
#include <wx/timer.h>
#include <queue>
#include "OVModelManager.h"

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
   void BeginInstallFor(ModelEntryPanel* panel, InstallQueueEntryPanel* queueEntry);
   InstallQueueEntryPanel* FindQueueEntry(ModelEntryPanel* panel) const;
   InstallQueueEntryPanel* FindQueueEntryById(int entryId) const;
   void RemoveQueueEntry(InstallQueueEntryPanel* entry);

   wxScrolledWindow* scrollPanel;
   wxBoxSizer* modelSizer;
   wxBoxSizer* queueSizer;
   wxTimer installTimer;

   std::vector<ModelEntryPanel*> allPanels;
   std::vector<InstallQueueEntryPanel*> queuePanels;
   std::queue<ModelEntryPanel*> installQueue;
   ModelEntryPanel* activeInstall = nullptr;
   InstallQueueEntryPanel* activeQueueEntry = nullptr;
   int installProgress = 0;

   wxDECLARE_EVENT_TABLE();
};

class ModelEntryPanel : public wxPanel {
public:
   ModelEntryPanel(wxWindow* parent, const std::string peffect, std::shared_ptr<OVModelManager::ModelInfo> minfo, ModelManagerDialog* manager, bool restartRequired);

   std::shared_ptr<OVModelManager::ModelInfo> GetModel() const { return model; }
   const std::string& GetEffect() const { return effect; }

   void UpdateStatus();
   void SetQueued();
   void SetInstalling();
   void SetInstalled();
   void SetFailed(const wxString& summary);

private:
   void OnInfo(wxCommandEvent& event);
   void OnInstall(wxCommandEvent& event);

   std::string effect;
   std::shared_ptr<OVModelManager::ModelInfo> model;
   ModelManagerDialog* manager;
   bool restartRequired{ false };

   wxButton* installButton;
};

class InstallQueueEntryPanel : public wxPanel {
public:
   InstallQueueEntryPanel(wxWindow* parent, ModelEntryPanel* sourcePanel);

   void SetAsInstalling();
   void SetAsQueued();
   void SetAsFailed(const OVModelManager::InstallResult& result);
   void UpdateProgress(int percent);
   ModelEntryPanel* GetSourcePanel() const;

private:
   void OnViewDetails(wxCommandEvent& event);

   ModelEntryPanel* modelPanel;
   wxStaticText* label;
   wxGauge* gauge;
   wxButton* detailsButton;
   OVModelManager::InstallResult installResult;
};

