// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#pragma once
#include <string>
#ifdef _WIN32 
#include <windows.h>
#else
#include <codecvt>
#include <locale>
#endif

static inline std::string FullPath(std::string base_dir, std::string filename)
{
#ifdef WIN32
   const std::string os_sep = "\\";
#else
   const std::string os_sep = "/";
#endif
   return base_dir + os_sep + filename;
}

static inline std::string wstring_to_string(const std::wstring& wstr) {
#ifdef _WIN32 
   int size_needed = WideCharToMultiByte(CP_ACP, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
   std::string strTo(size_needed, 0);
   WideCharToMultiByte(CP_ACP, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
   return strTo;
#else 
   std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_decoder;
   return wstring_decoder.to_bytes(wstr);
#endif 
}
