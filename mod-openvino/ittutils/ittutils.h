// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with ITT-related functions/macros/classes
 * @file ittutils.h
 */
#pragma once

#ifdef ITT_ENABLED
#include <ittnotify.h>
static __itt_domain* domain = __itt_domain_create("OMZ.ITT.Domain");

class ITTScopedTask
{
public:
    ITTScopedTask(const char* taskname)
        : handle(__itt_string_handle_create(taskname))
    {
        __itt_task_begin(domain, __itt_null, __itt_null, handle);
    };

    ~ITTScopedTask()
    {
        __itt_task_end(domain);
    }
private:
    __itt_string_handle* handle;
};

#define ITT_SCOPED_TASK(TASKNAME) ITTScopedTask itt_task_##TASKNAME(#TASKNAME);

#else
#define ITT_SCOPED_TASK(TASKNAME) {}
#endif
