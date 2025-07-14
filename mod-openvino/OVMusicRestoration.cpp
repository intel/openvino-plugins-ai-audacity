// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "OVMusicRestoration.h"
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

const ComponentInterfaceSymbol EffectOVMusicRestoration::Symbol{ XO("OpenVINO Music Restoration") };

namespace { BuiltinEffectsModule::Registration< EffectOVMusicRestoration > reg; }


EffectOVMusicRestoration::EffectOVMusicRestoration()
{

}

EffectOVMusicRestoration::~EffectOVMusicRestoration()
{

}

// ComponentInterface implementation
ComponentInterfaceSymbol EffectOVMusicRestoration::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVMusicRestoration::GetDescription() const
{
   return XO("Removes reverberation from spoken audio or vocals");
}

const std::string EffectOVMusicRestoration::ModelManagerName() const
{
   return OVModelManager::MusicRestorationName();
}

std::unordered_map<std::string, EffectOVDemixerEffect::SeparationModeEntry> EffectOVMusicRestoration::GetModelMap()
{
   std::unordered_map<std::string, EffectOVDemixerEffect::SeparationModeEntry> model_to_separation_map;

   {
      SeparationModeEntry entry;
      entry.stems = { "Restored" };
      entry.target_stem_for_instrumental = 0; //restored stem
      entry.bZeroPad = true;
      model_to_separation_map.emplace("Apollo MP3 Restore (@JusperLee)", entry);
   }

   {
      SeparationModeEntry entry;
      entry.stems = { "Restored" };
      entry.target_stem_for_instrumental = 0; //restored stem
      entry.bZeroPad = true;
      model_to_separation_map.emplace("Apollo Universal Restore (@Lew)", entry);
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
         + ", Only Reverb";

      std::cout << pair.first << ":" << std::endl;
      std::cout << "  " << all_stems_mode << std::endl;
      std::cout << "  " << instrumental_mode << std::endl;

      pair.second.guiSeparationModeSelections.push_back({ TranslatableString{ wxString(all_stems_mode), {}} });
      pair.second.guiSeparationModeSelections.push_back({ TranslatableString{ wxString(instrumental_mode), {}} });
   }

   return model_to_separation_map;
}
