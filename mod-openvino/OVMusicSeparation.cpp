// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "OVMusicSeparation.h"
#include "WaveTrack.h"
#include "EffectOutputTracks.h"
#include "effects/EffectEditor.h"
#include <math.h>
#include <iostream>
#include <wx/log.h>

#include "ViewInfo.h"
#include "TimeWarper.h"
#include "LoadEffects.h"
#include "htdemucs.h"

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

const ComponentInterfaceSymbol EffectOVMusicSeparation::Symbol{ XO("OpenVINO Music Separation") };

namespace { BuiltinEffectsModule::Registration< EffectOVMusicSeparation > reg; }

BEGIN_EVENT_TABLE(EffectOVMusicSeparation, wxEvtHandler)
EVT_CHECKBOX(ID_Type_AdvancedCheckbox, EffectOVMusicSeparation::OnAdvancedCheckboxChanged)
EVT_BUTTON(ID_Type_DeviceInfoButton, EffectOVMusicSeparation::OnDeviceInfoButtonClicked)
EVT_BUTTON(ID_Type_ModelManagerButton, EffectOVMusicSeparation::OnModelManagerButtonClicked)
END_EVENT_TABLE()

EffectOVMusicSeparation::EffectOVMusicSeparation()
{
   mSupportedDevices = ovdemucs::HTDemucs::GetSupportedDevices();

   ov::Core core;

   for (auto d : mSupportedDevices)
   {
      mGuiDeviceSelections.push_back({ TranslatableString{ wxString(d), {}} });

      m_simple_to_full_device_map.push_back({ d, core.get_property(d, "FULL_DEVICE_NAME").as<std::string>() });
   }

   mGuiSeparationModeSelections.push_back({ TranslatableString{ wxString("(2 Stem) Instrumental, Vocals"), {}} });
   mGuiSeparationModeSelections.push_back({ TranslatableString{ wxString("(4 Stem) Drums, Bass, Vocals, Others"), {}} });
}

EffectOVMusicSeparation::~EffectOVMusicSeparation()
{

}

// ComponentInterface implementation
ComponentInterfaceSymbol EffectOVMusicSeparation::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVMusicSeparation::GetDescription() const
{
   return XO("Splits a stereo track into 4 new tracks -- Bass, Drums, Vocals, Others");
}

VendorSymbol EffectOVMusicSeparation::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

unsigned EffectOVMusicSeparation::GetAudioInCount() const
{
   return 2;
}

// EffectDefinitionInterface implementation

EffectType EffectOVMusicSeparation::GetType() const
{
   return EffectTypeProcess;
}

bool EffectOVMusicSeparation::IsInteractive() const
{
   return true;
}

std::unique_ptr<EffectEditor> EffectOVMusicSeparation::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess&,
   const EffectOutputs*)
{
   mUIParent = S.GetParent();

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
         //m_deviceSelectionChoice
         mTypeChoiceSeparationModeCtrl = S.Id(ID_Type)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_separationModeSelectionChoice)
            .AddChoice(XXO("Separation Mode:"),
               Msgids(mGuiSeparationModeSelections.data(), mGuiSeparationModeSelections.size()));
      }
      S.EndMultiColumn();

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
         mNumberOfShiftsCtrl = S.Name(XO("Shifts"))
            .Validator<IntegerValidator<int>>(&mNumberOfShifts,
               NumValidatorStyle::DEFAULT,
               1,
               8)
            .AddTextBox(XO("Shifts"), L"", 12);

         advancedSizer = mNumberOfShiftsCtrl->GetContainingSizer();
      }
      S.EndMultiColumn();

   }
   S.EndVerticalLay();

   show_or_hide_advanced_options();

   return nullptr;
}

void EffectOVMusicSeparation::show_or_hide_advanced_options()
{
   if (advancedSizer)
   {
      advancedSizer->ShowItems(mbAdvanced);
      advancedSizer->Layout();
   }
}

void EffectOVMusicSeparation::OnAdvancedCheckboxChanged(wxCommandEvent& evt)
{
   mbAdvanced = mShowAdvancedOptionsCheckbox->GetValue();

   show_or_hide_advanced_options();

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

void EffectOVMusicSeparation::OnModelManagerButtonClicked(wxCommandEvent& evt)
{
   ShowModelManagerDialog();
}

void EffectOVMusicSeparation::OnDeviceInfoButtonClicked(wxCommandEvent& evt)
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
   EffectOVMusicSeparation* music_sep = (EffectOVMusicSeparation*)user;

   return music_sep->UpdateProgress(perc_complete);
}

bool EffectOVMusicSeparation::UpdateProgress(double perc)
{
   //TotalProgress will return true if user clicks 'cancel'
   if (TotalProgress(perc / 100.0))
   {
      std::cout << "User cancelled!" << std::endl;
      return false;
   }

   return true;

}

bool EffectOVMusicSeparation::Process(EffectInstance&, EffectSettings&)
{
   try
   {
      auto model_collection = OVModelManager::instance().GetModelCollection(OVModelManager::MusicSepName());

      // It shouldn't be possible for this condition to be true (User shoudn't have been able to click 'Apply'),
      // but double check anyway..
      if (!model_collection || model_collection->models.empty() || !model_collection->models[0]->installed)
      {
         throw std::runtime_error("Music Separation model has not been installed.");
      }

      FilePath model_folder = FileNames::MkDir(wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath());

      std::string demucs_v4_path = audacity::ToUTF8(wxFileName(model_collection->models[0]->installation_path, wxT("htdemucs_v4.xml"))
         .GetFullPath());

      FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());

      //Note: Using a variant of wstring conversion that seems to work more reliably when there are special characters present in the path.
      std::string cache_path = wstring_to_string(wxFileName(cache_folder).GetFullPath().ToStdWstring());

      std::cout << "demucs_v4_path = " << demucs_v4_path << std::endl;
      std::cout << "cache_path = " << cache_path << std::endl;
      std::cout << "number of shifts = " << mNumberOfShifts << std::endl;

      if (m_deviceSelectionChoice >= mSupportedDevices.size())
      {
         throw std::runtime_error("Invalid device choice id:  " +
            std::to_string(m_deviceSelectionChoice));
      }

      EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };

      bool bGoodResult = true;

      std::cout << "Creating OpenVINO-based HTDemucs object that will run on " << mSupportedDevices[m_deviceSelectionChoice] << std::endl;

      TotalProgress(0.01, XO("Compiling AI Model..."));

      std::shared_ptr< ovdemucs::HTDemucs > pHTDemucs;
      
      {
         auto device = mSupportedDevices[m_deviceSelectionChoice];
         auto create_htdemucs_fut = std::async(std::launch::async, [&demucs_v4_path, &device, &cache_path]() {

            // WA for OpenVINO locale caching issue (https://github.com/openvinotoolkit/openvino/issues/24370)
            OVLocaleWorkaround wa;
            return std::make_shared< ovdemucs::HTDemucs >(demucs_v4_path.c_str(), device, cache_path);
            });

         std::future_status status;
         float total_time = 0.f;
         do {
            using namespace std::chrono_literals;
            status = create_htdemucs_fut.wait_for(0.5s);
            {
               std::string message = "Loading Music Separation AI Model to " + device + "...";
               if (total_time > 10)
               {
                  message += " (This could take a while if this is the first time running this feature with this device)";
               }
               TotalProgress(0.01, TranslatableString{ wxString(message), {} });
            }

            total_time += 0.5;

         } while (status != std::future_status::ready);

         pHTDemucs = create_htdemucs_fut.get();
      }

      
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
            Floats entire_input{ total_samples * 2 };

            bool bOkay = pLeftChannel->GetFloats(entire_input.get(), start, total_samples);
            if (!bOkay)
            {
               throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
                  std::to_string(total_samples) + " samples");
            }

            bOkay = pRightChannel->GetFloats(entire_input.get() + total_samples, start, total_samples);
            if (!bOkay)
            {
               throw std::runtime_error("unable to get all right samples. GetFloats() failed for " +
                  std::to_string(total_samples) + " samples");
            }

            TotalProgress(0.01, XO("Applying Music Separation using OpenVINO"));

            //0: drums
            //1: bass
            //2: other instruments
            //3: vocals
            float* pOut[4];
            bool demucs_success = pHTDemucs->Apply(entire_input.get(),
               total_samples,
               pOut[0],
               pOut[1],
               pOut[2],
               pOut[3],
               mNumberOfShifts,
               HTDemucsProgressUpdate, this);

            if (!demucs_success)
            {
               return false;
            }

            auto pProject = FindProject();
            const auto& selectedRegion =
               ViewInfo::Get(*pProject).selectedRegion;

            std::vector<std::string> sourceLabels;
            if (m_separationModeSelectionChoice == 0)
            {
               sourceLabels = { "Instrumental", "Vocals" };

               // mix together drums, bass, and 'other instruments'.
               for (size_t i = 0; i < total_samples * 2; i++)
               {
                  pOut[0][i] = (pOut[0][i] + pOut[1][i] + pOut[2][i]) * pLeftChannel->GetChannelVolume(0);
               }

               // replace output index 1 with index 3 (vocals), so that we use proper channel
               // in the coming loop.
               pOut[1] = pOut[3];
            }
            else
            {
               sourceLabels = { "Drums", "Bass", "Other Instruments", "Vocals" };
            }

            auto orig_track_name = pTrack->GetName();
            for (int i = 0; i < sourceLabels.size(); i++)
            {
               // Workaround for 3.4.X issue where setting name of a new output track
               // retains the label of the track that it was copied from. So, we'll
               // change the name of the input track here, copy it, and then change it
               // back later.
               pTrack->SetName(orig_track_name + wxString("-" + sourceLabels[i]));

               //Create new output track from input track.
               auto newOutputTrack = pTrack->EmptyCopy();

               // create a temporary track list to append samples to
               auto pTmpTrack = pTrack->EmptyCopy();
               auto iter = pTmpTrack->Channels().begin();

               //append output samples to L & R channels.
               auto& tmpLeft = **iter++;
               tmpLeft.Append((samplePtr)pOut[i], floatSample, total_samples);

               if (pTrack->Channels().size() > 1)
               {
                  auto& tmpRight = **iter;
                  tmpRight.Append((samplePtr)(pOut[i] + total_samples), floatSample, total_samples);
               }

               //flush it
               pTmpTrack->Flush();

               // Clear & Paste into new output track
               newOutputTrack->ClearAndPaste(selectedRegion.t0() - pTmpTrack->GetStartTime(),
                  selectedRegion.t1() - pTmpTrack->GetStartTime(), *pTmpTrack);

               // TODO: For Audacity 3.4.X, this doesn't seem to work as expected.
               // The generated track will not have this name. Instead, it will retain
               // whatever the inputTrack's name was.
               //newOutputTrack->SetName(orig_track_name + wxString("-" + sourceLabels[i]));

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
      wxLogError("In Music Separation, exception: %s", error.what());
      EffectUIServices::DoMessageBox(*this,
         XO("Music Separation failed. See details in Help->Diagnostics->Show Log..."),
         wxICON_STOP,
         XO("Error"));
   }

   return false;
}

bool EffectOVMusicSeparation::TransferDataToWindow(const EffectSettings&)
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
   OVModelManager::instance().register_installed_callback(OVModelManager::MusicSepName(), callback);

   auto model_collection = OVModelManager::instance().GetModelCollection(OVModelManager::MusicSepName());
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
