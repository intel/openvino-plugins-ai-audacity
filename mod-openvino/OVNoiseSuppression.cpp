// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "OVNoiseSuppression.h"
#include "WaveTrack.h"
#include "EffectOutputTracks.h"
#include "effects/EffectEditor.h"
#include <math.h>
#include <iostream>

#include <wx/log.h>

#include "ViewInfo.h"
#include "TimeWarper.h"

#include <wx/intl.h>
#include <wx/valgen.h>

#include "ShuttleGui.h"

#include <wx/choice.h>
#include <wx/sizer.h>
#include <wx/checkbox.h>
#include <wx/stattext.h>

#include "FileNames.h"
#include "CodeConversions.h"

#include "LoadEffects.h"
#include <future>

#include "OVStringUtils.h"

#include <openvino/openvino.hpp>

#include "widgets/valnum.h"

#include "noise_suppression_omz_model.h"
#include "noise_suppression_df_model.h"

const ComponentInterfaceSymbol EffectOVNoiseSuppression::Symbol{ XO("OpenVINO Noise Suppression") };

namespace { BuiltinEffectsModule::Registration< EffectOVNoiseSuppression > reg; }

BEGIN_EVENT_TABLE(EffectOVNoiseSuppression, wxEvtHandler)
EVT_CHECKBOX(ID_Type_AdvancedCheckbox, EffectOVNoiseSuppression::OnAdvancedCheckboxChanged)
EVT_CHOICE(ID_Type_Model, EffectOVNoiseSuppression::OnAdvancedCheckboxChanged)
EVT_BUTTON(ID_Type_DeviceInfoButton, EffectOVNoiseSuppression::OnDeviceInfoButtonClicked)
END_EVENT_TABLE()

static bool is_deepfilter_model_present(std::string deepfilter_basename)
{
   auto model_folder = wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath();
   model_folder = wxFileName(model_folder, wxString(deepfilter_basename)).GetFullPath();

   std::vector< std::string > model_basenames = { "enc", "erb_dec", "df_dec" };
   for( auto mb : model_basenames)
   {
      auto binmodelpath = wxFileName(model_folder, wxString(mb + ".bin"));
      auto xmlmodelpath = wxFileName(model_folder, wxString(mb + ".xml"));

      if (!binmodelpath.FileExists())
      {
         std::cout << "is_deepfilter_model_present: returning false because " << mb + ".bin" << " doesn't exist." << std::endl;
         return false;
      }

      if (!xmlmodelpath.FileExists())
      {
         std::cout << "is_deepfilter_model_present: returning false because " << mb + ".xml" << " doesn't exist." << std::endl;
         return false;
      }
   }

   return true;
}

static bool is_omz_model_present(std::string omz_model_basename)
{
   auto model_folder = wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath();

   auto binmodelpath = wxFileName(model_folder, wxString(omz_model_basename + ".bin"));
   auto xmlmodelpath = wxFileName(model_folder, wxString(omz_model_basename + ".xml"));

   if (!binmodelpath.FileExists())
   {
      std::cout << "is_omz_model_present: returning false because " << omz_model_basename + ".bin" << " doesn't exist." << std::endl;
      return false;
   }

   if (!xmlmodelpath.FileExists())
   {
      std::cout << "is_omz_model_present: returning false because " << omz_model_basename + ".xml" << " doesn't exist." << std::endl;
      return false;
   }

   return true;
}


EffectOVNoiseSuppression::EffectOVNoiseSuppression()
{
   ov::Core core;

   auto ov_supported_device = core.get_available_devices();
   for (auto d : ov_supported_device)
   {
      //GNA devices are not supported
      if (d.find("GNA") != std::string::npos) continue;

      m_simple_to_full_device_map.push_back({ d, core.get_property(d, "FULL_DEVICE_NAME").as<std::string>() });

      mSupportedDevices.push_back(d);
      mGuiDeviceSelections.push_back({ TranslatableString{ wxString(d), {}} });
   }

   std::vector< std::string > deepfiltermodels = { "deepfilternet2", "deepfilternet3" };
   for (auto dfm : deepfiltermodels)
   {
      if (is_deepfilter_model_present(dfm))
      {
         mSupportedModels.push_back(dfm);
      }
   }

   std::vector< std::string > omzmodels = { "noise-suppression-denseunet-ll-0001" };
   for (auto omzm : omzmodels)
   {
      if (is_omz_model_present(omzm))
      {
         mSupportedModels.push_back(omzm);
      }
   }

   for (auto m : mSupportedModels)
   {
      mGuiModelSelections.push_back({ TranslatableString{ wxString(m), {}} });
   }
}

EffectOVNoiseSuppression::~EffectOVNoiseSuppression()
{

}

// ComponentInterface implementation
ComponentInterfaceSymbol EffectOVNoiseSuppression::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVNoiseSuppression::GetDescription() const
{
   return XO("Applies AI Background Noise Suppression using OpenVINO");
}

VendorSymbol EffectOVNoiseSuppression::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

// EffectDefinitionInterface implementation

EffectType EffectOVNoiseSuppression::GetType() const
{
   return EffectTypeProcess;
}

bool EffectOVNoiseSuppression::IsInteractive() const
{
   return true;
}

void EffectOVNoiseSuppression::show_or_hide_advanced_options()
{
   mbAdvanced = mShowAdvancedOptionsCheckbox->GetValue();

   if (attentuationLimitSizer)
   {
      int current_selection = mTypeChoiceModelCtrl->GetCurrentSelection();
      auto model_selection_string = audacity::ToUTF8(mTypeChoiceModelCtrl->GetString(current_selection));
      bool bDeepFilterModel = (model_selection_string == "deepfilternet2") || (model_selection_string == "deepfilternet3");

      attentuationLimitSizer->ShowItems(mbAdvanced && bDeepFilterModel);
      attentuationLimitSizer->Layout();
   }

   if (df3postfiltersizer)
   {
      int current_selection = mTypeChoiceModelCtrl->GetCurrentSelection();
      auto model_selection_string = audacity::ToUTF8(mTypeChoiceModelCtrl->GetString(current_selection));

      df3postfiltersizer->ShowItems(mbAdvanced && (model_selection_string == "deepfilternet3"));
      df3postfiltersizer->Layout();
   }

   if (noAdvancedSettingsLabel)
   {
      bool bShow = false;

      if (mbAdvanced)
      {
         bShow = !(attentuationLimitSizer->AreAnyItemsShown() || df3postfiltersizer->AreAnyItemsShown());
      }

      noAdvancedSettingsLabel->ShowItems(bShow);
      noAdvancedSettingsLabel->Layout();
   }
}

void EffectOVNoiseSuppression::OnAdvancedCheckboxChanged(wxCommandEvent& evt)
{
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

void EffectOVNoiseSuppression::OnDeviceInfoButtonClicked(wxCommandEvent& evt)
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

bool EffectOVNoiseSuppression::TransferDataToWindow(const EffectSettings&)
{
   if (!mUIParent || !mUIParent->TransferDataToWindow())
   {
      return false;
   }

   EffectEditor::EnablePreview(mUIParent, false);

   if (mSupportedModels.empty())
   {
      wxLogInfo("OpenVINO Noise Suppression has no models installed.");
      EffectEditor::EnableApply(mUIParent, false);
   }

   mDF3RunPostFilter->SetValue(mbRunDF3PostFilter);

   return true;
}

bool EffectOVNoiseSuppression::TransferDataFromWindow(EffectSettings&)
{
   if (!mUIParent->Validate() || !mUIParent->TransferDataFromWindow())
   {
      return false;
   }

   mbRunDF3PostFilter = mDF3RunPostFilter->GetValue();

   return true;
}

std::unique_ptr<EffectEditor> EffectOVNoiseSuppression::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess&,
   const EffectOutputs*)
{
   mUIParent = S.GetParent();

   S.AddSpace(0, 5);
   S.StartVerticalLay();
   {
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

      S.StartMultiColumn(4, wxLEFT);
      {
         //m_deviceSelectionChoice
         mTypeChoiceModelCtrl = S.Id(ID_Type_Model)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_modelSelectionChoice)
            .AddChoice(XXO("Noise Suppression Model:"),
               Msgids(mGuiModelSelections.data(), mGuiModelSelections.size()));

         mTypeChoiceModelCtrl->SetSelection(m_modelSelectionChoice);
      }
      S.EndMultiColumn();

      //advanced options
      S.StartMultiColumn(2, wxLEFT);
      {
         mShowAdvancedOptionsCheckbox = S.Id(ID_Type_AdvancedCheckbox).AddCheckBox(XXO("&Advanced Options"), mbAdvanced);
      }
      S.EndMultiColumn();

      S.StartStatic(XO("Attenuation Limit(dB)"));
      {
         S.AddVariableText(XO("100 means no attenuation limit (full noise suppression)\nFor little noise reduction, set to 6 - 12.\nFor medium, 18 - 24."));
         auto attn = S.Validator<FloatingPointValidator<float>>(
               6, &mAttenuationLimit,
               NumValidatorStyle::NO_TRAILING_ZEROES,
               0.0f,
               100.0f)
            .AddTextBox(XO(""), L"", 12);

         attentuationLimitSizer = attn->GetContainingSizer();
      }
      S.EndStatic();

      S.StartMultiColumn(1, wxLEFT);
      {
         auto text = S.AddVariableText(XO("No Advanced Options Available for this Model"));
         noAdvancedSettingsLabel = text->GetContainingSizer();
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         mDF3RunPostFilter = S.Id(ID_Type_AdvancedCheckbox).AddCheckBox(XXO("&Enable Post-filter that slightly over-attenuates very noisy sections."), mbRunDF3PostFilter);
         df3postfiltersizer = mDF3RunPostFilter->GetContainingSizer();
      }
      S.EndMultiColumn();
   }
   S.EndVerticalLay();

   show_or_hide_advanced_options();

   return nullptr;
}

bool EffectOVNoiseSuppression::UpdateProgress(double perc)
{
   return !TotalProgress(perc);
}

static bool NSProgressCallback(float perc_complete, void* user)
{
   EffectOVNoiseSuppression* pThis = (EffectOVNoiseSuppression*)user;

   return pThis->UpdateProgress(perc_complete);
}

bool EffectOVNoiseSuppression::Process(EffectInstance&, EffectSettings&)
{
   EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
   bool bGoodResult = true;

   ov::CompiledModel compiledModel;
   std::shared_ptr< NoiseSuppressionModel > ns_model;
   try
   {
      auto compile_compiledModel_fut = std::async(std::launch::async, [this, &compiledModel]() {
         std::shared_ptr< NoiseSuppressionModel > ret;
         try
         {
            //CompileNoiseSuppression(compiledModel);
            FilePath model_folder = FileNames::MkDir(wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath());
            FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());
            std::string cache_path = wstring_to_string(wxFileName(cache_folder).GetFullPath().ToStdWstring());

            // WA for OpenVINO locale caching issue (https://github.com/openvinotoolkit/openvino/issues/24370)
            OVLocaleWorkaround wa;

            auto model_selection_string = mSupportedModels[m_modelSelectionChoice];
            if ((model_selection_string == "deepfilternet2") || (model_selection_string == "deepfilternet3"))
            {
               ov_deepfilternet::ModelSelection dfnet_selection = ov_deepfilternet::ModelSelection::DEEPFILTERNET2;
               if (model_selection_string == "deepfilternet3")
               {
                  dfnet_selection = ov_deepfilternet::ModelSelection::DEEPFILTERNET3;
               }

               auto ns_df = std::make_shared< NoiseSuppressionDFModel >(audacity::ToUTF8(wxFileName(model_folder).GetFullPath()),
                  mSupportedDevices[m_deviceSelectionChoice], cache_path, dfnet_selection);

               std::cout << "setting attn limit of " << mAttenuationLimit << std::endl;
               ns_df->SetAttenLimit(mAttenuationLimit);

               std::cout << "setting df3 post filter to " << mbRunDF3PostFilter << std::endl;
               ns_df->SetDF3PostFilter(mbRunDF3PostFilter);

               ret = ns_df;
            }
            else
            {
               //must be an omz model then.
               auto model_file = model_selection_string + ".xml";
               std::string  model_path = audacity::ToUTF8(wxFileName(model_folder, wxString(model_file))
                  .GetFullPath());

               std::cout << "Using model path = " << model_path << std::endl;
               ret = std::make_shared< NoiseSuppressionOMZModel >(model_path, mSupportedDevices[m_deviceSelectionChoice], cache_path);
            }

            return ret;
         }
         catch (const std::exception& error) {
            wxLogError("In Noise Suppression Compilation, exception: %s", error.what());
            EffectUIServices::DoMessageBox(*this,
               XO("Noise Suppression failed. See details in Help->Diagnostics->Show Log..."),
               wxICON_STOP,
               XO("Error"));
            return ret;
         }
      });

      std::future_status status;
      float total_time = 0.f;
      do {
         using namespace std::chrono_literals;
         status = compile_compiledModel_fut.wait_for(0.5s);
         {
            std::string message = "Compiling Noise Suppression AI Model for " + mSupportedDevices[m_deviceSelectionChoice] + "...";
            if (total_time > 10)
            {
               message += " (This could take a while if this is the first time running this feature with this device)";
            }
            TotalProgress(0.01, TranslatableString{ wxString(message), {} });
         }

         total_time += 0.5;

      } while (status != std::future_status::ready);

      ns_model = compile_compiledModel_fut.get();

      if (!ns_model)
      {
         //It'd be kind of odd for control to reach here, as any compilation failure should
         // have thrown an exception which would put us in the below 'catch'.
         wxLogError("Noise Suppression Compilation appears to have failed.");
         EffectUIServices::DoMessageBox(*this,
            XO("Noise Suppression failed. See details in Help->Diagnostics->Show Log..."),
            wxICON_STOP,
            XO("Error"));
         return false;
      }
   }
   catch (const std::exception& error) {
      wxLogError("In Noise Suppression Compilation, exception: %s", error.what());
      EffectUIServices::DoMessageBox(*this,
         XO("Noise Suppression failed. See details in Help->Diagnostics->Show Log..."),
         wxICON_STOP,
         XO("Error"));
      return false;
   }

   std::cout << "Going into WaveTrack Selection loop!" << std::endl;

   try
   {
      //mCurTrackNum = 0;
      size_t wavetracki = 0;
      for (auto pOutWaveTrack : outputs.Get().Selected<WaveTrack>())
      {
         //Get start and end times from track
         double trackStart = pOutWaveTrack->GetStartTime();
         double trackEnd = pOutWaveTrack->GetEndTime();

         // Set the current bounds to whichever left marker is
         // greater and whichever right marker is less:
         const double curT0 = std::max(trackStart, mT0);
         const double curT1 = std::min(trackEnd, mT1);

         // Process only if the right marker is to the right of the left marker
         if (curT1 > curT0) {

            //first, copy the region of interest to a new track. We do this so that if we need
            // to downsample to meet model requirements, we don't affect the area outside of the
            // selection.
            auto pCopiedTrack = std::static_pointer_cast<WaveTrack>(pOutWaveTrack->Copy(curT0, curT1));

            double origRate = pCopiedTrack->GetRate();
            int model_sample_rate = ns_model->sample_rate();
            if (origRate != model_sample_rate)
            {
               std::cout << "resampling from " << origRate << " to " << model_sample_rate << std::endl;
               pCopiedTrack->Resample(model_sample_rate);
            }

            //Transform the marker timepoints to samples
            // Note, because of the above copy, we use start time of 0 here.
            auto start = pCopiedTrack->TimeToLongSamples(0);
            auto end = pCopiedTrack->TimeToLongSamples(curT1 - curT0);

            size_t total_samples = (end - start).as_size_t();

            for (size_t channeli = 0; channeli < pCopiedTrack->Channels().size(); channeli++)
            {
               std::string message = "Running Noise Suppression on Track " + std::to_string(wavetracki) + ", channel " + std::to_string(channeli);
               if (TotalProgress(0.01, TranslatableString{ wxString(message), {} }))
               {
                  return false;
               }

               auto pChannel = pCopiedTrack->GetChannel(channeli);

               if (!ns_model->run(pChannel, start, total_samples, NSProgressCallback, this) )
               {
                  return false;
               }
            }

            //resample back to original rate.
            if (origRate != model_sample_rate)
            {
               pCopiedTrack->Resample(origRate);
            }

            //now, paste 'filtered' samples stored in pCopiedTrack to output WaveTrack
            pOutWaveTrack->ClearAndPaste(curT0, curT1, *pCopiedTrack);
         }

         wavetracki++;
      }

      if (bGoodResult)
         outputs.Commit();

   }
   catch (const std::exception& error) {
      std::cout << "CompileNoiseSuppression routine failed: " << error.what() << std::endl;
      return false;
   }

   return bGoodResult;
}

