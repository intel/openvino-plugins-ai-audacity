# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-only

set( OV_MUSICGEN_SOURCES
musicgen/musicgen_decoder_model.h 
musicgen/music_gen_decoder_full_batch2.h 
musicgen/music_gen_decoder_full_batch1.h 
musicgen/music_gen_decoder_cl.h 
musicgen/music_gen_decoder_cl.cpp 
musicgen/musicgen_utils.h 
musicgen/encodec_encoder.h 
musicgen/musicgen_model_static.h
musicgen/musicgen_model_static.cpp
musicgen/musicgen_decoder.h
musicgen/musicgen_decoder.cpp
musicgen/static_kv_cache_manager_cl.h
musicgen/static_kv_cache_manager_cl.cpp
musicgen/musicgen_for_causal_lm.h
musicgen/musicgen_for_causal_lm.cpp
musicgen/musicgen_for_conditional_generation.cpp
musicgen/musicgen_for_conditional_generation.h
musicgen/musicgen_for_conditional_generation_refactor.cpp
musicgen/musicgen_for_conditional_generation_refactor.h
musicgen/musicgen.cpp
musicgen/musicgen.h
musicgen/musicgen_config.h 
PARENT_SCOPE)

