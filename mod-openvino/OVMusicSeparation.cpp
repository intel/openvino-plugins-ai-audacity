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

const ComponentInterfaceSymbol EffectOVMusicSeparation::Symbol{ XO("OpenVINO Music Separation") };

namespace { BuiltinEffectsModule::Registration< EffectOVMusicSeparation > reg; }


EffectOVMusicSeparation::EffectOVMusicSeparation()
{

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

const std::string EffectOVMusicSeparation::ModelManagerName() const
{
   return OVModelManager::MusicSepName();
}

std::unordered_map<std::string, EffectOVDemixerEffect::SeparationModeEntry> EffectOVMusicSeparation::GetModelMap()
{
   std::unordered_map<std::string, EffectOVDemixerEffect::SeparationModeEntry> model_to_separation_map;

   {
      SeparationModeEntry entry;
      entry.stems = { "Drums", "Bass", "Others", "Vocals" };
      entry.target_stem_for_instrumental = 3; //vocal stem
      model_to_separation_map.emplace("Demucs v4", entry);
   }

   {
      SeparationModeEntry entry;
      entry.stems = { "Drums", "dummy", "dummy", "dummy" };
      entry.target_stem_for_instrumental = 0; //drums stem
      model_to_separation_map.emplace("Demucs v4 FT Drums", entry);
   }

   {
      SeparationModeEntry entry;
      entry.stems = { "dummy", "Bass", "dummy", "dummy" };
      entry.target_stem_for_instrumental = 1; //bass stem
      model_to_separation_map.emplace("Demucs v4 FT Bass", entry);
   }

   {
      SeparationModeEntry entry;
      entry.stems = { "dummy", "dummy", "Other Instruments", "dummy" };
      entry.target_stem_for_instrumental = 2; //others stem
      model_to_separation_map.emplace("Demucs v4 FT Other Instruments", entry);
   }

   {
      SeparationModeEntry entry;
      entry.stems = { "dummy", "dummy", "dummy", "Vocals" };
      entry.target_stem_for_instrumental = 3; //vocal stem
      model_to_separation_map.emplace("Demucs v4 FT Vocals", entry);
   }

   {
      SeparationModeEntry entry;
      entry.stems = { "Drums", "Bass", "Others", "Vocals", "Guitar", "Piano" };
      entry.target_stem_for_instrumental = 3; //vocal stem
      model_to_separation_map.emplace("Demucs v4 6s", entry);
   }

   {
      SeparationModeEntry entry;
      entry.stems = { "Vocals" };
      entry.target_stem_for_instrumental = 0; //vocal stem
      model_to_separation_map.emplace("MelBandRoformer Vocals (Kimberly Jenson version)", entry);
   }

   {
      SeparationModeEntry entry;
      entry.stems = { "Crowd" };
      entry.target_stem_for_instrumental = 0; //crowd stem
      model_to_separation_map.emplace("MelBandRoformer Crowd", entry);
   }

   for (auto& pair : model_to_separation_map)
   {
      // Count number of non-dummy stems
      int non_dummy_stems = 0;
      for (auto& s : pair.second.stems)
         if (s != "dummy") non_dummy_stems++;

      // Generate the 'all stems' selection string.
      std::string all_stems_mode = "(" + std::to_string(non_dummy_stems) + " Stem) ";
      int stems_added = 1;
      for (auto& s : pair.second.stems)
      {
         if (s != "dummy")
         {
            all_stems_mode += s;
            if (stems_added < non_dummy_stems)
               all_stems_mode += ", ";
            stems_added++;
         }
      }

      if (pair.second.target_stem_for_instrumental >= pair.second.stems.size())
      {
         throw std::runtime_error("GetModelMap: pair.second.target_stem_for_instrumental >= pair.second.stems.size");
      }
      std::string instrumental_mode = "(2 Stem) " + pair.second.stems[pair.second.target_stem_for_instrumental]
         + ", Instrumental";

      std::cout << pair.first << ":" << std::endl;
      std::cout << "  " << all_stems_mode << std::endl;
      std::cout << "  " << instrumental_mode << std::endl;

      pair.second.guiSeparationModeSelections.push_back({ TranslatableString{ wxString(all_stems_mode), {}} });
      pair.second.guiSeparationModeSelections.push_back({ TranslatableString{ wxString(instrumental_mode), {}} });
   }

   return model_to_separation_map;
}
