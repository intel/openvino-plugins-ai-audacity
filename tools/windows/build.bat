setlocal

IF NOT EXIST env.bat (
    echo env.bat is not set. Exiting.
    exit /b 1
)

call env.bat

echo LIBTORCH_DIR=%LIBTORCH_DIR%
echo OPENVINO_GENAI_DIR=%OPENVINO_GENAI_DIR%
echo AUDACITY_CLONE_DIR=%AUDACITY_CLONE_DIR%
echo BUILD_FOLDER=%BUILD_FOLDER%
echo CONAN_HOME=%CONAN_HOME%

echo Path=%Path%

:: Copyright (C) 2024 Intel Corporation
:: SPDX-License-Identifier: GPL-3.0-only

IF "%OPENVINO_GENAI_DIR%"=="" (
    echo OPENVINO_GENAI_DIR is not set. Exiting.
    exit /b 1
)

IF "%LIBTORCH_DIR%"=="" (
    echo LIBTORCH_DIR is not set. Exiting.
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

set "bat_path=%~dp0"
set audacity_add_ov_mod_patch_path=%bat_path%add_ov_module.patch
set audacity_no_vc_runtime_install_patch=%bat_path%audacity_no_vc_runtime_install.patch

:: Set up OpenVINO GenAI build environment.
call %OPENVINO_GENAI_DIR%\setupvars.bat || exit /b 1

:: Setup Libtorch end.
set LIBTORCH_ROOTDIR=%LIBTORCH_DIR%
set Path=%LIBTORCH_ROOTDIR%\lib;%Path%

::::::::::::::::::::::::::::::::::::::::::::
:: Audacity  + OpenVINO AI Plugins build. ::
::::::::::::::::::::::::::::::::::::::::::::
IF NOT EXIST %AUDACITY_CLONE_DIR% (
    echo Can't find audacity source directory.
    echo /B 1
)


set current_work_dir=%cd%
:: apply patch that adds mod-openvino to build
cd %AUDACITY_CLONE_DIR%

:: Check if 'git' command exists
git --version >nul 2>&1
IF NOT ERRORLEVEL 1 (
  echo Applying patch using git command...
  git apply --ignore-whitespace %audacity_add_ov_mod_patch_path% || exit /b 1
  git apply --ignore-whitespace %audacity_no_vc_runtime_install_patch% || exit /b 1
) ELSE (
  :: Since git is not available, check if 'patch' command exists
  patch --version >nul 2>&1
  IF NOT ERRORLEVEL 1 (
    echo Applying patch using patch command...
    patch -p1 < %audacity_add_ov_mod_patch_path% || exit /b 1
    patch -p1 < %audacity_no_vc_runtime_install_patch% || exit /b 1
  ) ELSE (
    echo Neither git nor patch command is available.
    exit /b 1
  )
)

cd %current_work_dir%
set current_work_dir=

xcopy %AI_PLUGIN_REPO_SOURCE_FOLDER%mod-openvino "%AUDACITY_CLONE_DIR%\modules\etc\mod-openvino" /E /I || exit /b 1

:: Build Audacity + our OpenVINO module
mkdir audacity-build
cd audacity-build


if defined Python3_ROOT_DIR (
    set Python3_ROOT_DIR_DEFINE=-DPython3_ROOT_DIR=%Python3_ROOT_DIR%
) else (
    set Python3_ROOT_DIR_DEFINE=""
)

:: Run cmake
cmake %AUDACITY_CLONE_DIR% -DCMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_NO_WARNINGS=TRUE -DAUDACITY_BUILD_LEVEL=%AUDACITY_BUILD_LEVEL% %Python3_ROOT_DIR_DEFINE% -DCMAKE_DISABLE_FIND_PACKAGE_MKL=TRUE -DCMAKE_DISABLE_FIND_PACKAGE_MKLDNN=TRUE -Daudacity_has_networking=ON || exit /b 1

:: build it
cmake --build . --config %AUDACITY_BUILD_CONFIG% || exit /b 1
cd ..

endlocal