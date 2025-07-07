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
