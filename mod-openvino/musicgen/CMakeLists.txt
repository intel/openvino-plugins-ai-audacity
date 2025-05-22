# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-only

set( OV_MUSICGEN_SOURCES
musicgen/musicgen_utils.h 
musicgen/encodec_encoder.h 
musicgen/musicgen_decoder.h
musicgen/musicgen_decoder.cpp
musicgen/static_kv_cache_manager_cl.h
musicgen/static_kv_cache_manager_cl.cpp
musicgen/musicgen_for_conditional_generation_refactor.cpp
musicgen/musicgen_for_conditional_generation_refactor.h
musicgen/musicgen.cpp
musicgen/musicgen.h
musicgen/musicgen_config.h 
PARENT_SCOPE)

