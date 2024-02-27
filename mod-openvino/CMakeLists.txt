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

set( SOURCES
      OVNoiseSuppression.cpp
      OVNoiseSuppression.h
      OVMusicSeparation.cpp
      OVMusicSeparation.h
      OVMusicGeneration.cpp
      OVMusicGeneration.h
      OVMusicStyleRemix.cpp
      OVMusicStyleRemix.h
      OVWhisperTranscription.cpp
      OVWhisperTranscription.h
      htdemucs.cpp
      htdemucs.h
      OpenVINO.cpp
      OVStringUtils.h
)

set( LIBRARIES
   PRIVATE
      Audacity
      openvino::runtime
      ${TORCH_LIBRARIES}
      ${whisper}
      ${stable_diffusion_ov}
      ${stable_diffusion_audio_ov}
)

audacity_module( ${TARGET} "${SOURCES}" "${LIBRARIES}" "" "" )

if(MSVC)
   target_compile_options(${TARGET} PRIVATE
   /analyze /sdl /Gy /DYNAMICBASE /GS
   )
endif()


