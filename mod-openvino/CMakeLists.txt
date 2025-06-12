# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-only

set( TARGET mod-openvino )

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenVINOGenAI REQUIRED)

message("ENV LIBTORCH_ROOTDIR = $ENV{LIBTORCH_ROOTDIR}")

set(LIBTORCH_ROOTDIR $ENV{LIBTORCH_ROOTDIR})

set(CMAKE_PREFIX_PATH "${LIBTORCH_ROOTDIR}/share/cmake/Torch")

find_package(Torch REQUIRED)
message("torch libraries = ${TORCH_LIBRARIES}")

#for musicgen
#whether ITT is enabled or not, include ittutils.
# This allows the project source files to unconditionally include <ittutils.h>
# That header file will implement the preprocessor logic to *actually* enable ITT or not.
include_directories( ittutils )

add_subdirectory(musicgen)
include_directories(musicgen)

set( NOISE_SUPPRESSION_SOURCES
      OVNoiseSuppression.cpp
      OVNoiseSuppression.h
      noise_suppression/noise_suppression_model.h
      noise_suppression/noise_suppression_omz_model.h
      noise_suppression/noise_suppression_omz_model.cpp
      noise_suppression/noise_suppression_df_model.h
      noise_suppression/noise_suppression_df_model.cpp
      noise_suppression/deepfilternet/deepfilter.cpp
      noise_suppression/deepfilternet/deepfilter.h
      noise_suppression/deepfilternet/multiframe.h
      noise_suppression/deepfilternet/multiframe.cpp
      noise_suppression/deepfilternet/dfnet_model.h
      noise_suppression/deepfilternet/dfnet_model.cpp
)

include_directories(noise_suppression)
include_directories(noise_suppression/deepfilternet)

set( AUDIO_SR_SOURCES 
     OVAudioSR.cpp
     OVAudioSR.h
     audio_sr/audiosr_common.h
     audio_sr/autoencoder.cpp
     audio_sr/autoencoder.h
     audio_sr/ddim_sampler.cpp
     audio_sr/ddim_sampler.h
     audio_sr/ddpm_latent_diffusion.cpp
     audio_sr/ddpm_latent_diffusion.h
     audio_sr/distributions.h
     audio_sr/audio_sr.h
     audio_sr/audio_sr.cpp
)

include_directories(audio_sr)

set( SOURCES
      ${NOISE_SUPPRESSION_SOURCES}
      ${AUDIO_SR_SOURCES}
      OVMusicSeparation.cpp
      OVMusicSeparation.h
      OVMusicGenerationLLM.cpp
      OVMusicGenerationLLM.h
      OVWhisperTranscriptionGenAI.cpp
      OVWhisperTranscriptionGenAI.h
      htdemucs.cpp
      htdemucs.h
      ${OV_MUSICGEN_SOURCES}
      OpenVINO.cpp
      OVStringUtils.h
      OVModelManagerUI.h
      OVModelManagerUI.cpp
      OVModelManager.h
      OVModelManager.cpp
      OVModelManagerPopulation.cpp
)

set( LIBRARIES
   PRIVATE
      Audacity
      lib-network-manager
      openvino::runtime
      openvino::genai
      ${TORCH_LIBRARIES}
)

audacity_module( ${TARGET} "${SOURCES}" "${LIBRARIES}" "" "" )

if(MSVC)
   target_compile_options(${TARGET} PRIVATE
   /analyze /sdl /Gy /DYNAMICBASE /GS /D_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING /D_CRT_SECURE_NO_WARNINGS
   )
endif()


