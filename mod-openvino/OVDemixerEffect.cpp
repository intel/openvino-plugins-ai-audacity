// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "OVDemixerEffect.h"

#include "WaveTrack.h"
#include "EffectOutputTracks.h"
#include "effects/EffectEditor.h"
#include <math.h>
#include <iostream>
#include <wx/log.h>

#include "ViewInfo.h"
#include "TimeWarper.h"
#include "LoadEffects.h"

#include <wx/intl.h>
#include <wx/valgen.h>
#include <wx/checkbox.h>
#include <wx/sizer.h>

#include "ShuttleGui.h"
#include <wx/choice.h>
#include "FileNames.h"
#include "CodeConversions.h"
#include <future>

#include "widgets/valnum.h"

#include "OVStringUtils.h"
#include <openvino/openvino.hpp>
#include "OVModelManager.h"
#include "OVModelManagerUI.h"

#include "demix/demix.h"
#include "demix/htdemucs.h"
#include "demix/mel_band_roformer.h"
#include "demix/apollo.h"

BEGIN_EVENT_TABLE(EffectOVDemixerEffect, wxEvtHandler)
EVT_CHECKBOX(ID_Type_AdvancedCheckbox, EffectOVDemixerEffect::OnAdvancedCheckboxChanged)
EVT_BUTTON(ID_Type_DeviceInfoButton, EffectOVDemixerEffect::OnDeviceInfoButtonClicked)
EVT_BUTTON(ID_Type_ModelManagerButton, EffectOVDemixerEffect::OnModelManagerButtonClicked)
EVT_CHOICE(ID_Type_ModelSelection, EffectOVDemixerEffect::OnModelSelectionChanged)
END_EVENT_TABLE()


bool EffectOVDemixerEffect::Init()
{
    if (_bInitAlreadySuccessful)
        return true;

    try
    {
        m_effectName = GetSymbol().Internal().ToStdString();
        m_modelManagerName = ModelManagerName();

        ov::Core core;
        auto ov_supported_device = core.get_available_devices();
        for (auto d : ov_supported_device)
        {
            m_simple_to_full_device_map.push_back({ d, core.get_property(d, "FULL_DEVICE_NAME").as<std::string>() });

            mSupportedDevices.push_back(d);
            mGuiDeviceSelections.push_back({ TranslatableString{ wxString(d), {}} });
        }

        mGuiSeparationModeSelections.push_back({ TranslatableString{ wxString(" "), {}} });
        mGuiSeparationModeSelections.push_back({ TranslatableString{ wxString(" "), {}} });

        m_model_to_separation_modes = GetModelMap();

        _bInitAlreadySuccessful = true;
    }
    catch (const std::exception& error)
    {
        wxLogError("In %s Init, exception: %s", m_effectName, error.what());
        return false;
    }

    return true;
}

VendorSymbol EffectOVDemixerEffect::GetVendor() const
{
    return XO("OpenVINO AI Effects");
}

unsigned EffectOVDemixerEffect::GetAudioInCount() const
{
    return 2;
}

// EffectDefinitionInterface implementation

EffectType EffectOVDemixerEffect::GetType() const
{
    return EffectTypeProcess;
}

bool EffectOVDemixerEffect::IsInteractive() const
{
    return true;
}

std::unique_ptr<EffectEditor> EffectOVDemixerEffect::PopulateOrExchange(
    ShuttleGui& S, EffectInstance&, EffectSettingsAccess&,
    const EffectOutputs*)
{
    mUIParent = S.GetParent();

    auto collection = OVModelManager::instance().GetModelCollection(m_modelManagerName);
    for (auto& model_info : collection->models) {
        if (model_info->installed) {
            if (std::find(mSupportedModels.begin(), mSupportedModels.end(), model_info->model_name) == mSupportedModels.end()) {
                mSupportedModels.push_back(model_info->model_name);
            }
        }
    }

    mGuiModelSelections.clear();
    for (auto& m : mSupportedModels)
    {
        mGuiModelSelections.push_back({ TranslatableString{ wxString(m), {}} });
    }

    OVModelManager::InstalledCallback callback =
        [this](const std::string& model_name) {
        wxTheApp->CallAfter([=]() {
            if (std::find(mSupportedModels.begin(), mSupportedModels.end(), model_name) == mSupportedModels.end()) {
                mSupportedModels.push_back(model_name);
                mGuiModelSelections.push_back({ TranslatableString{ wxString(model_name), {}} });
            }

            if (mUIParent)
            {
                EffectEditor::EnableApply(mUIParent, true);
                EffectEditor::EnablePreview(mUIParent, false);
                if (mTypeChoiceModelSelection)
                {
                    mTypeChoiceModelSelection->Append(wxString(model_name));

                    if (mTypeChoiceModelSelection->GetCount() == 1) {
                        mTypeChoiceModelSelection->SetSelection(mTypeChoiceModelSelection->GetCount() - 1);
                    }
                }
            }
            });
        };

    OVModelManager::instance().register_installed_callback(m_modelManagerName, callback);

    S.AddSpace(0, 5);
    S.StartVerticalLay();
    {
        S.StartMultiColumn(1, wxLEFT);
        {
            auto model_manager_button = S.Id(ID_Type_ModelManagerButton).AddButton(XO("Open Model Manager"));
        }
        S.EndMultiColumn();

        S.StartMultiColumn(2, wxLEFT);
        {
            mTypeChoiceModelSelection = S.Id(ID_Type_ModelSelection)
                .MinSize({ -1, -1 })
                .Validator<wxGenericValidator>(&m_modelSelectionChoice)
                .AddChoice(XXO("Model Selection:"),
                    Msgids(mGuiModelSelections.data(), mGuiModelSelections.size()));
        }
        S.EndMultiColumn();

        S.StartMultiColumn(2, wxLEFT);
        {
            //m_deviceSelectionChoice
            mTypeChoiceSeparationModeCtrl = S.Id(ID_Type)
                .MinSize({ -1, -1 })
                .Validator<wxGenericValidator>(&m_separationModeSelectionChoice)
                .AddChoice(XXO("Separation Mode:"),
                    Msgids(mGuiSeparationModeSelections.data(), mGuiSeparationModeSelections.size()));
        }
        S.EndMultiColumn();

        SetModelSeparationModeSelections();

        S.StartStatic(XO(""), wxLEFT);
        {
            S.StartMultiColumn(4, wxEXPAND);
            {
                mTypeChoiceDeviceCtrl = S.Id(ID_Type)
                    .MinSize({ -1, -1 })
                    .Validator<wxGenericValidator>(&m_deviceSelectionChoice)
                    .AddChoice(XXO("OpenVINO Inference Device:"),
                        Msgids(mGuiDeviceSelections.data(), mGuiDeviceSelections.size()));
                S.AddVariableText(XO(""));

                auto device_info_button = S.Id(ID_Type_DeviceInfoButton).AddButton(XO("Device Details..."));

                S.SetStretchyCol(2);
            }
            S.EndMultiColumn();
        }
        S.EndStatic();

        //advanced options
        S.StartMultiColumn(2, wxLEFT);
        {
            mShowAdvancedOptionsCheckbox = S.Id(ID_Type_AdvancedCheckbox).AddCheckBox(XXO("&Advanced Options"), false);
        }
        S.EndMultiColumn();

        S.StartMultiColumn(2, wxLEFT);
        {
            mNumberOfShiftsCtrl = S.Name(XO("Overlaps"))
                .Validator<IntegerValidator<int>>(&mNumberOfShifts,
                    NumValidatorStyle::DEFAULT,
                    1,
                    8)
                .AddTextBox(XO("Overlaps"), L"", 12);

            advancedSizer = mNumberOfShiftsCtrl->GetContainingSizer();
        }
        S.EndMultiColumn();

    }
    S.EndVerticalLay();

    show_or_hide_advanced_options();

    return nullptr;
}

void EffectOVDemixerEffect::show_or_hide_advanced_options()
{
    if (advancedSizer)
    {
        advancedSizer->ShowItems(mbAdvanced);
        advancedSizer->Layout();
    }
}

void EffectOVDemixerEffect::FitWindowToCorrectSize()
{
    if (mUIParent)
    {
        mUIParent->Layout();
        mUIParent->SetMinSize(mUIParent->GetSizer()->GetMinSize());
        mUIParent->SetSize(mUIParent->GetSizer()->GetMinSize());
        mUIParent->Fit();
        auto p = mUIParent->GetParent();
        if (p)
        {
            p->Fit();
        }
    }
}

void EffectOVDemixerEffect::OnAdvancedCheckboxChanged(wxCommandEvent& evt)
{
    mbAdvanced = mShowAdvancedOptionsCheckbox->GetValue();

    show_or_hide_advanced_options();

    FitWindowToCorrectSize();
}

void EffectOVDemixerEffect::OnModelManagerButtonClicked(wxCommandEvent& evt)
{
    ShowModelManagerDialog();
}

void EffectOVDemixerEffect::OnDeviceInfoButtonClicked(wxCommandEvent& evt)
{
    std::string device_mapping_str = "";
    for (auto e : m_simple_to_full_device_map)
    {
        device_mapping_str += e.first + " = " + e.second;
        device_mapping_str += "\n";
    }
    auto v = TranslatableString(device_mapping_str, {});

    EffectUIServices::DoMessageBox(*this,
        v,
        wxICON_INFORMATION,
        XO("OpenVINO Device Details"));
}

void EffectOVDemixerEffect::SetModelSeparationModeSelections()
{
    if (!mTypeChoiceSeparationModeCtrl || !mTypeChoiceModelSelection)
        return;

    int current_selection = mTypeChoiceModelSelection->GetCurrentSelection();
    if (current_selection == -1)
    {
        current_selection = m_modelSelectionChoice;
    }

    auto selected_model = audacity::ToUTF8(mTypeChoiceModelSelection->GetString(current_selection));

    auto it = m_model_to_separation_modes.find(selected_model);
    if (it == m_model_to_separation_modes.end())
    {
        wxLogError("SetModelSeparationModeSelections: No model entry for selected_model=", selected_model);
        return;
    }

    // First, clear the list
    mTypeChoiceSeparationModeCtrl->Clear();

    for (auto& mode_selection : it->second.guiSeparationModeSelections)
    {
        mTypeChoiceSeparationModeCtrl->AppendString(mode_selection.StrippedTranslation());
    }

    mTypeChoiceSeparationModeCtrl->SetSelection(m_separationModeSelectionChoice);

    FitWindowToCorrectSize();
}

void EffectOVDemixerEffect::OnModelSelectionChanged(wxCommandEvent& evt)
{
    // reset separation mode choice to the first entry in the list.
    m_separationModeSelectionChoice = 0;

    SetModelSeparationModeSelections();
}

static std::vector<WaveTrack::Holder> CreateSourceTracks
(WaveTrack* leader, std::vector<std::string>& labels)
{
    std::vector<WaveTrack::Holder> sources;
    for (auto& label : labels)
    {
        WaveTrack::Holder srcTrack = leader->EmptyCopy();

        srcTrack->SetSelected(false);
        srcTrack->SetSolo(false);

        // append the source name to the track's name
        srcTrack->SetName(srcTrack->GetName() + wxString("-" + label));
        sources.emplace_back(srcTrack);
    }
    return sources;
}

static bool HTDemucsProgressUpdate(double perc_complete, void* user)
{
   EffectOVDemixerEffect* demixer = (EffectOVDemixerEffect*)user;

    perc_complete = perc_complete * 100;
    if (perc_complete > 100)
        perc_complete = 100;

    return demixer->UpdateProgress(perc_complete);
}

bool EffectOVDemixerEffect::UpdateProgress(double perc)
{
    //TotalProgress will return true if user clicks 'cancel'
    if (TotalProgress(perc / 100.0))
    {
        std::cout << "User cancelled!" << std::endl;
        return false;
    }

    return true;
}

bool EffectOVDemixerEffect::Process(EffectInstance&, EffectSettings&)
{
    try
    {
        std::string model_selection_str = mSupportedModels[m_modelSelectionChoice];

        auto model_collection = OVModelManager::instance().GetModelCollection(m_modelManagerName);

        // It shouldn't be possible for this condition to be true (User shoudn't have been able to click 'Apply'),
        // but double check anyway..
        if (!model_collection || model_collection->models.empty())
        {
            throw std::runtime_error(m_effectName + " models have not been installed.");
        }

        std::shared_ptr< OVModelManager::ModelInfo > retrieved_model_info;
        for (auto model_info : model_collection->models) {
            if (model_info && model_info->installed && model_info->model_name == model_selection_str)
            {
                retrieved_model_info = model_info;
            }
        }

        if (!retrieved_model_info) {
            throw std::runtime_error("Couldn't retrieve installed model info for " + model_selection_str);
        }

        if (!retrieved_model_info->installed) {
            throw std::runtime_error("This model is not installed: " + retrieved_model_info->model_name);
        }

        auto model_folder = retrieved_model_info->installation_path;

        auto sep_mode_it = m_model_to_separation_modes.find(model_selection_str);
        if (sep_mode_it == m_model_to_separation_modes.end())
        {
            throw std::runtime_error("Did not find separation mode entry for model_selection_str=" + model_selection_str);
        }

        std::shared_ptr< ov_demix::DemixModel > model;
        {
            auto device = mSupportedDevices[m_deviceSelectionChoice];
            FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());

            //Note: Using a variant of wstring conversion that seems to work more reliably when there are special characters present in the path.
            std::string cache_path = wstring_to_string(wxFileName(cache_folder).GetFullPath().ToStdWstring());

            std::cout << "model_folder = " << model_folder << std::endl;
            std::cout << "cache_path = " << cache_path << std::endl;
            std::cout << "number of shifts = " << mNumberOfShifts << std::endl;

            auto create_model_fut = std::async(std::launch::async, [&model_folder, &device, &cache_path, &model_selection_str, &sep_mode_it]()
                {
                    // WA for OpenVINO locale caching issue (https://github.com/openvinotoolkit/openvino/issues/24370)
                    OVLocaleWorkaround wa;

                    std::shared_ptr< ov_demix::DemixModel > ret;
                    if (model_selection_str.find("Demucs") != std::string::npos)
                    {
                        ret = std::make_shared<ov_demix::HTDemucs>(model_folder, device, cache_path);
                    }
                    else if (model_selection_str.find("MelBandRoformer") != std::string::npos)
                    {
                        auto pad_mode = ov_demix::DemixModel::PadMode::Reflect;

                        if (sep_mode_it->second.bZeroPad)
                        {
                            pad_mode = ov_demix::DemixModel::PadMode::Constant0;
                        }

                        ret = std::make_shared<ov_demix::MelBandRoformer>(model_folder, device, cache_path, pad_mode);
                    }
                    else if (model_selection_str.find("Apollo") != std::string::npos)
                    {
                        ret = std::make_shared<ov_demix::Apollo>(model_folder, device, cache_path);
                    }
                    else
                    {
                        throw std::runtime_error("Only HTDemucs or MelBandRoformer models are supported right now. ");
                    }
                    return ret;

                });

            std::future_status status;
            float total_time = 0.f;
            do {
                using namespace std::chrono_literals;
                status = create_model_fut.wait_for(0.5s);
                {
                    std::string message = "Loading " + model_selection_str + " AI Model to " + device + "...";
                    if (total_time > 10)
                    {
                        message += " (This could take a while if this is the first time running this feature with this device)";
                    }
                    TotalProgress(0.01, TranslatableString{ wxString(message), {} });
                }

                total_time += 0.5;

            } while (status != std::future_status::ready);

            model = create_model_fut.get();

            if (!model)
            {
                throw std::runtime_error("Error loading model to device...");
            }

        }

        auto stem_labels = sep_mode_it->second.stems;

        if (m_deviceSelectionChoice >= mSupportedDevices.size())
        {
            throw std::runtime_error("Invalid device choice id:  " +
                std::to_string(m_deviceSelectionChoice));
        }

        EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };

        bool bGoodResult = true;

        std::cout << "Creating OpenVINO-based HTDemucs object that will run on " << mSupportedDevices[m_deviceSelectionChoice] << std::endl;

        TotalProgress(0.01, XO("Compiling AI Model..."));

        std::vector< WaveTrack::Holder > tracks_to_process;
        std::vector< int > orig_rates;

        //Create resampled copies of the selected portion of tracks.
        // This prevents the 'Resample' operation to modify the user's
        // original track.
        for (auto track : outputs.Get().Selected<WaveTrack>())
        {
            auto left = track->GetChannel(0);
            auto start = left->GetStartTime();
            auto end = left->GetEndTime();
            if (track->Channels().size() > 1)
            {
                auto right = track->GetChannel(1);
                start = wxMin(start, right->GetStartTime());
                end = wxMax(end, right->GetEndTime());
            }

            // Set the current bounds to whichever left marker is
            // greater and whichever right marker is less:
            const double curT0 = std::max(start, mT0);
            const double curT1 = std::min(end, mT1);

            auto start_s = left->TimeToLongSamples(curT0);
            auto end_s = left->TimeToLongSamples(curT1);

            size_t total_samples = (end_s - start_s).as_size_t();
            Floats entire_input{ total_samples };

            // create a temporary track list to append samples to
            auto pTmpTrack = track->EmptyCopy();

            bool bOkay = left->GetFloats(entire_input.get(), start_s, total_samples);
            if (!bOkay)
            {
                throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
                    std::to_string(total_samples) + " samples");
            }

            auto iter =
                pTmpTrack->Channels().begin();


            //append output samples to L & R channels.
            auto& tmpLeft = **iter++;
            tmpLeft.Append((samplePtr)entire_input.get(), floatSample, total_samples);

            if (track->Channels().size() > 1)
            {
                auto right = track->GetChannel(1);
                bOkay = right->GetFloats(entire_input.get(), start_s, total_samples);
                if (!bOkay)
                {
                    throw std::runtime_error("unable to get all right samples. GetFloats() failed for " +
                        std::to_string(total_samples) + " samples");
                }

                auto& tmpRight = **iter;
                tmpRight.Append((samplePtr)entire_input.get(), floatSample, total_samples);
            }

            //flush it
            pTmpTrack->Flush();

            orig_rates.push_back(pTmpTrack->GetRate());
            if (pTmpTrack->GetRate() != 44100)
            {
                pTmpTrack->Resample(44100, mProgress);
            }

            tracks_to_process.push_back(pTmpTrack);
        }

        for (size_t ti = 0; ti < tracks_to_process.size(); ti++)
        {
            auto pTrack = tracks_to_process[ti];

            auto pLeftChannel = pTrack->GetChannel(0);
            auto pRightChannel = pLeftChannel;
            if (pTrack->Channels().size() > 1)
            {
                pRightChannel = pTrack->GetChannel(1);
            }

            double trackStart = pTrack->GetStartTime();
            double trackEnd = pTrack->GetEndTime();

            // Set the current bounds to whichever left marker is
            // greater and whichever right marker is less:
            const double curT0 = trackStart;
            const double curT1 = trackEnd;

            if (curT1 > curT0) {
                auto start = pTrack->TimeToLongSamples(curT0);
                auto end = pTrack->TimeToLongSamples(curT1);

                //Get the length of the buffer (as double). len is
                //used simple to calculate a progress meter, so it is easier
                //to make it a double now than it is to do it later
                auto len = (end - start).as_double();

                size_t total_samples = (end - start).as_size_t();

                auto left_samples = std::make_shared<std::vector<float>>(total_samples);
                auto right_samples = std::make_shared<std::vector<float>>(total_samples);

                bool bOkay = pLeftChannel->GetFloats(left_samples->data(), start, total_samples);
                if (!bOkay)
                {
                    throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
                        std::to_string(total_samples) + " samples");
                }

                bOkay = pRightChannel->GetFloats(right_samples->data(), start, total_samples);
                if (!bOkay)
                {
                    throw std::runtime_error("unable to get all right samples. GetFloats() failed for " +
                        std::to_string(total_samples) + " samples");
                }

                TotalProgress(0.01, TranslatableString{ wxString("Applying " + m_effectName), {} });

                ov_demix::AudioTrack input_channels = { left_samples, right_samples };
                auto output_stems = ov_demix::Demix(model, input_channels, mNumberOfShifts, HTDemucsProgressUpdate, this);

                auto pProject = FindProject();
                const auto& selectedRegion =
                    ViewInfo::Get(*pProject).selectedRegion;

                if (output_stems.size() != stem_labels.size())
                {
                    throw std::runtime_error("Expected Demix to return " + std::to_string(stem_labels.size()) +
                        " stems, but instead it returned " + std::to_string(output_stems.size()) + " stems.");
                }

                // m_separationModeSelectionChoice==1 means instrumental mode is being used.
                if (m_separationModeSelectionChoice == 1)
                {
                    auto target_stem_for_instrumental = sep_mode_it->second.target_stem_for_instrumental;
                    if (target_stem_for_instrumental >= output_stems.size())
                    {
                        throw std::runtime_error("Specified target_stem_for_instrumental out of range");
                    }

                    auto target_stem = output_stems[target_stem_for_instrumental];
                    ov_demix::GenerateInstrumental(target_stem, input_channels);

                    // replace output stems vector
                    std::vector<ov_demix::AudioTrack> new_output_stems = { target_stem, input_channels };
                    output_stems = new_output_stems;

                    // replace stem_labels vector
                    stem_labels = { sep_mode_it->second.stems[target_stem_for_instrumental], "Instrumental" };
                }

                auto orig_track_name = pTrack->GetName();
                for (int i = 0; i < stem_labels.size(); i++)
                {
                    // If it's a dummy stem, skip it.
                    if (stem_labels[i] == "dummy") continue;

                    auto& stem_track = output_stems[i];

                    // Workaround for 3.4.X issue where setting name of a new output track
                    // retains the label of the track that it was copied from. So, we'll
                    // change the name of the input track here, copy it, and then change it
                    // back later.
                    pTrack->SetName(orig_track_name + wxString(" - " + model_selection_str + " - " + stem_labels[i]));

                    //Create new output track from input track.
                    auto newOutputTrack = pTrack->EmptyCopy();

                    // create a temporary track list to append samples to
                    auto pTmpTrack = pTrack->EmptyCopy();
                    auto iter = pTmpTrack->Channels().begin();

                    //append output samples to L & R channels.
                    auto& tmpLeft = **iter++;
                    tmpLeft.Append((samplePtr)stem_track.first->data(), floatSample, total_samples);

                    if (pTrack->Channels().size() > 1)
                    {
                        auto& tmpRight = **iter;
                        tmpRight.Append((samplePtr)stem_track.second->data(), floatSample, total_samples);
                    }

                    //flush it
                    pTmpTrack->Flush();

                    // Clear & Paste into new output track
                    newOutputTrack->ClearAndPaste(selectedRegion.t0() - pTmpTrack->GetStartTime(),
                        selectedRegion.t1() - pTmpTrack->GetStartTime(), *pTmpTrack);

                    //Resample to original tracks rate
                    newOutputTrack->Resample(orig_rates[ti]);

                    if (TracksBehaviorsSolo.ReadEnum() == SoloBehaviorSimple)
                    {
                        //If in 'simple' mode, if original track is solo,
                        // mute the new track and set it to *not* be solo.
                        if (newOutputTrack->GetSolo())
                        {
                            newOutputTrack->SetMute(true);
                            newOutputTrack->SetSolo(false);
                        }
                    }

                    // Add the new track list to the output.
                    outputs.AddToOutputTracks(std::move(newOutputTrack));

                    // Change name back to original. Part of workaround described above.
                    pTrack->SetName(orig_track_name);
                }
            }
            else
            {
                throw std::runtime_error("unexpected case encountered where curT0 (" + std::to_string(curT0) +
                    ") <= curT1(" + std::to_string(curT1) + ")");
            }
        }

        if (bGoodResult)
            outputs.Commit();

        return bGoodResult;
    }
    catch (const std::exception& error) {
       std::string message = m_effectName + " failed. See details in Help->Diagnostics->Show Log...";
        wxLogError("In %s, exception: %s", m_effectName, error.what());
        EffectUIServices::DoMessageBox(*this,
           TranslatableString{ wxString(message), {} },
            wxICON_STOP,
            XO("Error"));
    }

    return false;
}

bool EffectOVDemixerEffect::TransferDataToWindow(const EffectSettings&)
{
    if (!mUIParent || !mUIParent->TransferDataToWindow())
    {
        return false;
    }

    OVModelManager::InstalledCallback callback =
        [this](const std::string& model_name) {
        wxTheApp->CallAfter([=]() {
            EffectEditor::EnableApply(mUIParent, true);
            EffectEditor::EnablePreview(mUIParent, false);
            });
        };
    OVModelManager::instance().register_installed_callback(m_modelManagerName, callback);

    auto model_collection = OVModelManager::instance().GetModelCollection(m_modelManagerName);
    if (!model_collection || model_collection->models.empty() || !model_collection->models[0]->installed)
    {
        EffectEditor::EnableApply(mUIParent, false);
    }
    else
    {
        EffectEditor::EnableApply(mUIParent, true);
    }
    EffectEditor::EnablePreview(mUIParent, false);

    return true;
}
