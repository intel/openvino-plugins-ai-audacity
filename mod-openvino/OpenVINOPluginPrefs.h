// OpenVINOPluginSettings.h
#pragma once

#include <wx/string.h>

namespace OpenVINOPluginSettings
{
   // Returns a directory that exists; if unset, defaults to DataDir()/openvino-models and creates it.
   wxString GetOrCreateModelDir(bool persistIfMissing = true);

   // Returns a directory that exists; if unset, defaults to DataDir()/openvino-model-cache and creates it.
   wxString GetOrCreateCompiledModelCacheDir(bool persistIfMissing = true);

   // Read/write helpers
   bool ReadEnableCache(bool defaultValue = true);
}
