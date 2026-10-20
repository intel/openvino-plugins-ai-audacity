// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include "StatefulEffect.h"
#include "effects/StatefulEffectUIServices.h"
#include "ShuttleAutomation.h"
#include <wx/weakref.h>
#include <mutex>

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
    bool SaveSettingsAsString(
        const EffectSettings& settings, wxString& parms) const override;
    [[nodiscard]] OptionalMessage LoadSettingsFromString(
        const wxString& parms, EffectSettings& settings) const override;
    bool SaveSettings(
        const EffectSettings& settings, CommandParameters& parms) const override;
    bool LoadSettings(
        const CommandParameters& parms, EffectSettings& settings) const override;

    OptionalMessage LoadUserPreset(
        const RegistryPath& name, EffectSettings& settings) const override;
    bool SaveUserPreset(
        const RegistryPath& name, const EffectSettings& settings) const override;

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

       std::string instrumental_name = "Instrumental";

       // For some models (i.e. crowd removal), it makes sense to
       // *only* provide the instrumental option.
       bool bOnlyInstrumental = false;

       //TODO: Move this someplace else. It's specific to MelBandRoformer models.
       bool bZeroPad = false;
    };

    virtual std::unordered_map<std::string, SeparationModeEntry> GetModelMap() = 0;

    bool m_bDisplaySepMode = true;

private:

    wxChoice* mTypeChoiceDeviceCtrl;
    int m_deviceSelectionChoice = 0;

    enum control
    {
        ID_Type = 10000,
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

    int mNumberOfOverlaps = 2;
    wxTextCtrl* mNumberOfOverlapsCtrl = nullptr;

    std::vector<std::pair<std::string, std::string>> m_simple_to_full_device_map;

    std::unordered_map< std::string, SeparationModeEntry > m_model_to_separation_modes;

    void SetModelSeparationModeSelections();
    void ClampSelectionChoices();
    void FitWindowToCorrectSize();

    std::string m_effectName = "Unknown Effect";
    std::string m_modelManagerName = "Unknown Name";

    bool _bInitAlreadySuccessful = false;

    std::mutex mProgMutex;
    double mProgressFrac = 0.;
    bool mIsCancelled = false;

    DECLARE_EVENT_TABLE()
};
