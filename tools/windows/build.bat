@echo off
setlocal

:: Copyright (C) 2024 Intel Corporation
:: SPDX-License-Identifier: GPL-3.0-only

:: Repo's we will clone from.
set OPENVINO_AUDACITY_AI_REPO_CLONE_URL=https://github.com/intel/openvino-plugins-ai-audacity.git
set AUDACITY_REPO_CLONE_URL=https://github.com/audacity/audacity.git
set WHISPERCPP_REPO_CLONE_URL=https://github.com/ggerganov/whisper.cpp

IF "%BUILD_FOLDER%"=="" (
    echo BUILD_FOLDER is not set. Exiting.
    exit /b
)

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

IF "%AUDACITY_REPO_CHECKOUT%"=="" (
    echo AUDACITY_REPO_CHECKOUT is not set. Exiting.
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

IF "%OPENVINO_AUDACITY_AI_REPO_CHECKOUT%"=="" (
    echo OPENVINO_AUDACITY_AI_REPO_CHECKOUT is not set. Exiting.
    exit /b
)

IF "%WHISPERCPP_REPO_CHECKOUT%"=="" (
    echo WHISPERCPP_REPO_CHECKOUT is not set. Exiting.
    exit /b
)

set "bat_path=%~dp0"
set "audacity_add_ov_mod_patch_path=%bat_path%add_ov_module.patch

echo "audacity_add_ov_mod_patch_path=%audacity_add_ov_mod_patch_path%"

:: go into build folder
cd %BUILD_FOLDER%

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
git clone --depth 1 --branch %WHISPERCPP_REPO_CHECKOUT% %WHISPERCPP_REPO_CLONE_URL%

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

:: Create local python env, just to install conan.
python -m venv build_env

echo "activating..."
call "build_env\Scripts\activate"

echo "installing conan"
pip install conan


echo "Cloning OpenVINO AI Plugins for Audacity"

:: clone OpenVINO AI Plugins for Audacity
git clone --depth 1 --branch %OPENVINO_AUDACITY_AI_REPO_CHECKOUT% %OPENVINO_AUDACITY_AI_REPO_CLONE_URL%

echo "Cloning Audacity"

:: clone Audacity
git clone --depth 1 --branch %AUDACITY_REPO_CHECKOUT% %AUDACITY_REPO_CLONE_URL%

:: apply patch that adds mod-openvino to build
cd audacity
git apply %audacity_add_ov_mod_patch_path%
cd ..

echo "Copying mod-openvino into audacity\modules"
xcopy "openvino-plugins-ai-audacity\mod-openvino" "audacity\modules\mod-openvino" /E /I

:: Build Audacity + our OpenVINO module
mkdir audacity-build
cd audacity-build

:: Run cmake
cmake ..\audacity -DAUDACITY_BUILD_LEVEL=%AUDACITY_BUILD_LEVEL%

:: build it
cmake --build . --config %AUDACITY_BUILD_CONFIG%
cd ..

endlocal