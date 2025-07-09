// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "StatefulEffect.h"
#include "effects/StatefulEffectUIServices.h"
#include <wx/weakref.h>

class WaveTrack;
class wxChoice;
class wxCheckBox;
class wxTextCtrl;
class wxSizer;

class EffectOVDemixerEffect : public StatefulEffect, public StatefulEffectUIServices
{
public:
    bool Init() override;

    VendorSymbol GetVendor() const override;

    unsigned GetAudioInCount() const override;

    // EffectDefinitionInterface implementation

    EffectType GetType() const override;
    bool IsInteractive() const override;

    bool Process(EffectInstance& instance, EffectSettings& settings) override;

    bool UpdateProgress(double perc);

    std::unique_ptr<EffectEditor> PopulateOrExchange(
        ShuttleGui& S, EffectInstance& instance,
        EffectSettingsAccess& access, const EffectOutputs* pOutputs) override;
    bool TransferDataToWindow(const EffectSettings& settings) override;

    void OnAdvancedCheckboxChanged(wxCommandEvent& evt);
    void OnDeviceInfoButtonClicked(wxCommandEvent& evt);
    void OnModelSelectionChanged(wxCommandEvent& evt);
    void OnModelManagerButtonClicked(wxCommandEvent& evt);

protected:

    virtual const std::string ModelManagerName() const = 0;

    struct SeparationModeEntry
    {
       int separationModeSelectionChoice = 0;
       std::vector< EnumValueSymbol > guiSeparationModeSelections;

       std::vector<std::string> stems;
       int target_stem_for_instrumental = 0;
    };

    virtual std::unordered_map<std::string, SeparationModeEntry> GetModelMap() = 0;

private:

    wxChoice* mTypeChoiceDeviceCtrl;
    int m_deviceSelectionChoice = 0;

    enum control
    {
        ID_Type = 10000,
        ID_Type_AdvancedCheckbox,
        ID_Type_DeviceInfoButton,
        ID_Type_ModelManagerButton,
        ID_Type_ModelSelection
    };

    std::vector< std::string > mSupportedModels;
    std::vector< EnumValueSymbol > mGuiModelSelections;
    int m_modelSelectionChoice = 0;
    wxChoice* mTypeChoiceModelSelection;

    std::vector< std::string > mSupportedDevices;
    std::vector< EnumValueSymbol > mGuiDeviceSelections;

    wxChoice* mTypeChoiceSeparationModeCtrl;
    int m_separationModeSelectionChoice = 0;

    // There are dummy values only used for wxChoice creation.
    std::vector< EnumValueSymbol > mGuiSeparationModeSelections;

    wxWeakRef<wxWindow> mUIParent{};

    wxCheckBox* mShowAdvancedOptionsCheckbox;

    int mNumberOfShifts = 1;
    wxTextCtrl* mNumberOfShiftsCtrl = nullptr;

    void show_or_hide_advanced_options();
    wxSizer* advancedSizer = nullptr;
    bool mbAdvanced = false;

    std::vector<std::pair<std::string, std::string>> m_simple_to_full_device_map;

    std::unordered_map< std::string, SeparationModeEntry > m_model_to_separation_modes;

    void SetModelSeparationModeSelections();
    void FitWindowToCorrectSize();

    std::string m_effectName = "Unknown Effect";
    std::string m_modelManagerName = "Unknown Name";

    DECLARE_EVENT_TABLE()
};
