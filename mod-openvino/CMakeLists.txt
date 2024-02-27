# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-only

set( TARGET mod-openvino )

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

message("ENV LIBTORCH_ROOTDIR = $ENV{LIBTORCH_ROOTDIR}")

set(LIBTORCH_ROOTDIR $ENV{LIBTORCH_ROOTDIR})

set(CMAKE_PREFIX_PATH "${LIBTORCH_ROOTDIR}/share/cmake/Torch")

find_package(Torch REQUIRED)
message("torch libraries = ${TORCH_LIBRARIES}")

set(WHISPERCPP_ROOTDIR $ENV{WHISPERCPP_ROOTDIR})
find_library(whisper NAMES whisper HINTS ${WHISPERCPP_ROOTDIR}/lib/static ${WHISPERCPP_ROOTDIR}/lib )
include_directories(${WHISPERCPP_ROOTDIR}/include )
message("whisper library = ${whisper}")

set(CPP_STABLE_DIFFUSION_OV_ROOTDIR $ENV{CPP_STABLE_DIFFUSION_OV_ROOTDIR})
find_library(stable_diffusion_ov NAMES stable_diffusion_ov HINTS ${CPP_STABLE_DIFFUSION_OV_ROOTDIR}/lib/static ${CPP_STABLE_DIFFUSION_OV_ROOTDIR}/lib )
find_library(stable_diffusion_audio_ov NAMES stable_diffusion_audio_ov HINTS ${CPP_STABLE_DIFFUSION_OV_ROOTDIR}/lib/static ${CPP_STABLE_DIFFUSION_OV_ROOTDIR}/lib )
include_directories(${CPP_STABLE_DIFFUSION_OV_ROOTDIR}/include )

message("stable_diffusion_ov library = ${stable_diffusion_ov}")
message("stable_diffusion_audio_ov library = ${stable_diffusion_audio_ov}")

#for musicgen
#whether ITT is enabled or not, include ittutils.
# This allows the project source files to unconditionally include <ittutils.h>
# That header file will implement the preprocessor logic to *actually* enable ITT or not.
include_directories( ittutils )

find_package(OpenCL REQUIRED)

add_subdirectory(musicgen)
include_directories(musicgen)


set( SOURCES
      OVNoiseSuppression.cpp
      OVNoiseSuppression.h
      OVMusicSeparation.cpp
      OVMusicSeparation.h
      #OVMusicGeneration.cpp
      #OVMusicGeneration.h
      OVMusicGenerationV2.cpp
      OVMusicGenerationV2.h
      #OVMusicStyleRemix.cpp
      #OVMusicStyleRemix.h
      OVWhisperTranscription.cpp
      OVWhisperTranscription.h
      htdemucs.cpp
      htdemucs.h
      ${OV_MUSICGEN_SOURCES}
      OpenVINO.cpp
      OVStringUtils.h
)

set( LIBRARIES
   PRIVATE
      Audacity
      openvino::runtime
	  OpenCL::OpenCL
      ${TORCH_LIBRARIES}
      ${whisper}
      #${stable_diffusion_ov}
      #${stable_diffusion_audio_ov}
)

audacity_module( ${TARGET} "${SOURCES}" "${LIBRARIES}" "" "" )

if(MSVC)
   target_compile_options(${TARGET} PRIVATE
   /analyze /sdl /Gy /DYNAMICBASE /GS /D _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
   )
endif()


