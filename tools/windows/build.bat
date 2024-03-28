@echo off
setlocal

:: Copyright (C) 2024 Intel Corporation
:: SPDX-License-Identifier: GPL-3.0-only
IF "%OPENVINO_DIR%"=="" (
    echo OPENVINO_DIR is not set. Exiting.
    exit /b
)

IF "%OPENVINO_TOKENIZERS_DIR%"=="" (
    echo OPENVINO_TOKENIZERS_DIR is not set. Exiting.
    exit /b
)

IF "%LIBTORCH_DIR%"=="" (
    echo LIBTORCH_DIR is not set. Exiting.
    exit /b
)

IF "%AUDACITY_BUILD_LEVEL%"=="" (
    echo AUDACITY_BUILD_LEVEL is not set. Exiting.
    exit /b
)

IF "%AUDACITY_BUILD_CONFIG%"=="" (
    echo AUDACITY_BUILD_CONFIG is not set. Exiting.
    exit /b
)

IF "%AI_PLUGIN_REPO_SOURCE_FOLDER%"=="" (
    echo AI_PLUGIN_REPO_SOURCE_FOLDER is not set. Exiting.
    exit /b
)

set "bat_path=%~dp0"
set "audacity_add_ov_mod_patch_path=%bat_path%add_ov_module.patch

echo "audacity_add_ov_mod_patch_path=%audacity_add_ov_mod_patch_path%"

:: Set up OpenVINO build environment.
call %OPENVINO_DIR%\setupvars.bat

:: Setup Libtorch end.
set LIBTORCH_ROOTDIR=%LIBTORCH_DIR%
set Path=%LIBTORCH_ROOTDIR%\lib;%Path%

:: Setup OpenCL env.
set OCL_ROOT=%OPENCL_SDK_DIR%
set Path=%OCL_ROOT%\bin;%Path%

::::::::::::::::::::::::
:: Whisper.cpp build. ::
::::::::::::::::::::::::

IF NOT EXIST whisper.cpp (
    echo Can't find whisper.cpp directory.
    echo /B
)

:: Create build folder
mkdir whisper-build-avx
cd whisper-build-avx

:: Run CMake, specifying that you want to enable OpenVINO support.
cmake ..\whisper.cpp -A x64 -DWHISPER_OPENVINO=ON

:: Build it:
cmake --build . --config Release

:: Install built whisper collateral into a local 'installed' directory:
cmake --install . --config Release --prefix .\installed

:: Setup whisper.cpp env.
set WHISPERCPP_ROOTDIR=%cd%\installed
set Path=%WHISPERCPP_ROOTDIR%\bin;%Path%

cd ..

:: Also build the non-AVX version
mkdir whisper-build-no-avx
cd whisper-build-no-avx

:: Run CMake, specifying that you want to enable OpenVINO support, but no AVX / AVX2 / other advanced instruction support
cmake ..\whisper.cpp -A x64 -DWHISPER_OPENVINO=ON -DWHISPER_NO_AVX=ON -DWHISPER_NO_AVX2=ON -DWHISPER_NO_FMA=ON -DWHISPER_NO_F16C=ON

:: Build it:
cmake --build . --config Release

:: Install built whisper collateral into a local 'installed' directory:
cmake --install . --config Release --prefix .\installed
cd ..

::::::::::::::::::::::::::::::::::::::::::::
:: Audacity  + OpenVINO AI Plugins build. ::
::::::::::::::::::::::::::::::::::::::::::::
IF NOT EXIST audacity (
    echo Can't find whisper.cpp directory.
    echo /B
)
:: apply patch that adds mod-openvino to build
cd audacity
git apply %audacity_add_ov_mod_patch_path%
cd ..

echo "Copying mod-openvino into audacity\modules"
xcopy %AI_PLUGIN_REPO_SOURCE_FOLDER%mod-openvino "audacity\modules\mod-openvino" /E /I

:: Build Audacity + our OpenVINO module
mkdir audacity-build
cd audacity-build

:: Run cmake
cmake ..\audacity -DAUDACITY_BUILD_LEVEL=%AUDACITY_BUILD_LEVEL%

:: build it
cmake --build . --config %AUDACITY_BUILD_CONFIG%
cd ..

endlocal