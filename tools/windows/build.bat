setlocal

IF NOT EXIST env.bat (
    echo env.bat is not set. Exiting.
    exit /b 1
)

call env.bat

echo LIBTORCH_DIR=%LIBTORCH_DIR%
echo OPENVINO_DIR=%OPENVINO_DIR%
echo OPENVINO_TOKENIZERS_DIR=%OPENVINO_TOKENIZERS_DIR%
echo OPENCL_SDK_DIR=%OPENCL_SDK_DIR%
echo WHISPER_CLONE_DIR=%WHISPER_CLONE_DIR%
echo AUDACITY_CLONE_DIR=%AUDACITY_CLONE_DIR%
echo BUILD_FOLDER=%BUILD_FOLDER%
echo CONAN_HOME=%CONAN_HOME%

echo Path=%Path%

:: Copyright (C) 2024 Intel Corporation
:: SPDX-License-Identifier: GPL-3.0-only
IF "%OPENVINO_DIR%"=="" (
    echo OPENVINO_DIR is not set. Exiting.
    exit /b 1
)

IF "%OPENVINO_TOKENIZERS_DIR%"=="" (
    echo OPENVINO_TOKENIZERS_DIR is not set. Exiting.
    exit /b 1
)

IF "%LIBTORCH_DIR%"=="" (
    echo LIBTORCH_DIR is not set. Exiting.
    exit /b 1
)

IF "%OPENCL_SDK_DIR%"=="" (
    echo OPENCL_SDK_DIR is not set. Exiting.
    exit /b 1
)

IF "%AUDACITY_BUILD_LEVEL%"=="" (
    echo AUDACITY_BUILD_LEVEL is not set. Exiting.
    exit /b 1
)

IF "%AUDACITY_BUILD_CONFIG%"=="" (
    echo AUDACITY_BUILD_CONFIG is not set. Exiting.
    exit /b 1
)

IF "%AI_PLUGIN_REPO_SOURCE_FOLDER%"=="" (
    echo AI_PLUGIN_REPO_SOURCE_FOLDER is not set. Exiting.
    exit /b 1
)

IF "%AUDACITY_CLONE_DIR%"=="" (
    echo AUDACITY_CLONE_DIR is not set. Exiting.
    exit /b 1
)

IF "%WHISPER_CLONE_DIR%"=="" (
    echo WHISPER_CLONE_DIR is not set. Exiting.
    exit /b 1
)


set "bat_path=%~dp0"
set "audacity_add_ov_mod_patch_path=%bat_path%add_ov_module.patch

echo audacity_add_ov_mod_patch_path=%audacity_add_ov_mod_patch_path%

:: Set up OpenVINO build environment.
call %OPENVINO_DIR%\setupvars.bat || exit /b 1

:: Setup Libtorch end.
set LIBTORCH_ROOTDIR=%LIBTORCH_DIR%
set Path=%LIBTORCH_ROOTDIR%\lib;%Path%

:: Setup OpenCL env.
set OCL_ROOT=%OPENCL_SDK_DIR%
set Path=%OCL_ROOT%\bin;%Path%

::::::::::::::::::::::::
:: Whisper.cpp build. ::
::::::::::::::::::::::::

IF NOT EXIST %WHISPER_CLONE_DIR% (
    echo Can't find whisper.cpp directory.
    echo /B 1
)

:: Create build folder
mkdir whisper-build-avx
cd whisper-build-avx

:: Run CMake, specifying that you want to enable OpenVINO support.
cmake %WHISPER_CLONE_DIR% -A x64 -DWHISPER_OPENVINO=ON || exit /b 1

:: Build it:
cmake --build . --config Release || exit /b 1

:: Install built whisper collateral into a local 'installed' directory:
cmake --install . --config Release --prefix .\installed || exit /b 1

:: Setup whisper.cpp env.
set WHISPERCPP_ROOTDIR=%cd%\installed
set Path=%WHISPERCPP_ROOTDIR%\bin;%Path%

cd ..

:: Also build the non-AVX version
mkdir whisper-build-no-avx
cd whisper-build-no-avx

:: Run CMake, specifying that you want to enable OpenVINO support, but no AVX / AVX2 / other advanced instruction support
cmake %WHISPER_CLONE_DIR%  -A x64 -DWHISPER_OPENVINO=ON -DWHISPER_NO_AVX=ON -DWHISPER_NO_AVX2=ON -DWHISPER_NO_FMA=ON -DWHISPER_NO_F16C=ON || exit /b 1

:: Build it:
cmake --build . --config Release || exit /b 1

:: Install built whisper collateral into a local 'installed' directory:
cmake --install . --config Release --prefix .\installed || exit /b 1
cd ..

::::::::::::::::::::::::::::::::::::::::::::
:: Audacity  + OpenVINO AI Plugins build. ::
::::::::::::::::::::::::::::::::::::::::::::
IF NOT EXIST %AUDACITY_CLONE_DIR% (
    echo Can't find whisper.cpp directory.
    echo /B 1
)

set current_work_dir=%cd%
:: apply patch that adds mod-openvino to build
cd %AUDACITY_CLONE_DIR%

:: Check if 'git' command exists
git --version >nul 2>&1
IF NOT ERRORLEVEL 1 (
  echo Applying patch using git command...
  git apply %audacity_add_ov_mod_patch_path% || exit /b 1
) ELSE (
  :: Since git is not available, check if 'patch' command exists
  patch --version >nul 2>&1
  IF NOT ERRORLEVEL 1 (
    echo Applying patch using patch command...
    patch -p1 < %audacity_add_ov_mod_patch_path% || exit /b 1
  ) ELSE (
    echo Neither git nor patch command is available.
    exit /b 1
  )
)

cd %current_work_dir%
set current_work_dir=

xcopy %AI_PLUGIN_REPO_SOURCE_FOLDER%mod-openvino "%AUDACITY_CLONE_DIR%\modules\mod-openvino" /E /I || exit /b 1

:: Build Audacity + our OpenVINO module
mkdir audacity-build
cd audacity-build

:: Run cmake
cmake %AUDACITY_CLONE_DIR% -DAUDACITY_BUILD_LEVEL=%AUDACITY_BUILD_LEVEL% || exit /b 1

:: build it
cmake --build . --config %AUDACITY_BUILD_CONFIG% || exit /b 1
cd ..

endlocal