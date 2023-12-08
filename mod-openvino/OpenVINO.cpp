// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "ModuleConstants.h"
#include <iostream>

//DEFINE_MODULE_ENTRIES


DEFINE_VERSION_CHECK

extern "C" DLL_API int ModuleDispatch(ModuleDispatchTypes type) {
   return 1;
}
