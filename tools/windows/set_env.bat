@echo off

set "bat_path=%~dp0"

set PATH_TO_SET=

:: Check if exactly 4 arguments are passed
if "%~1"=="" goto error
if "%~2"=="" goto error
if "%~3"=="" goto error
if "%~4"=="" goto error
if "%~5"=="" goto error

:: Build Level 0=alpha, 1=beta, 2=release
set AUDACITY_BUILD_LEVEL=2
set AUDACITY_BUILD_CONFIG=RelWithDebInfo

:: The version that we will pass to inno setup as the app version.
set AI_PLUGIN_VERSION=v3.6.4-R3.4

set AI_PLUGIN_REPO_SOURCE_FOLDER=%bat_path%\..\..\
echo AI_PLUGIN_REPO_SOURCE_FOLDER=%AI_PLUGIN_REPO_SOURCE_FOLDER%

call :ConvertRelToAbsPath %1 || exit /b 1
set LIBTORCH_ABS_PATH=%ABS_PATH%

call :ConvertRelToAbsPath %2 || exit /b 1
set OPENVINO_ABS_PATH=%ABS_PATH%

call :ConvertRelToAbsPath %3 || exit /b 1
set OPENCL_SDK_ABS_PATH=%ABS_PATH%

call :ConvertRelToAbsPath %4 || exit /b 1
set WHISPER_CLONE_ABS_PATH=%ABS_PATH%

call :ConvertRelToAbsPath %5 || exit /b 1
set AUDACITY_CLONE_ABS_PATH=%ABS_PATH%

set LIBTORCH_DIR=%LIBTORCH_ABS_PATH%
set OPENVINO_DIR=%OPENVINO_ABS_PATH%
set OPENCL_SDK_DIR=%OPENCL_SDK_ABS_PATH%
set WHISPER_CLONE_DIR=%WHISPER_CLONE_ABS_PATH%
set AUDACITY_CLONE_DIR=%AUDACITY_CLONE_ABS_PATH%

:: print some env.
echo LIBTORCH_DIR=%LIBTORCH_DIR%
echo OPENVINO_DIR=%OPENVINO_DIR%
echo OPENCL_SDK_DIR=%OPENCL_SDK_DIR%
echo WHISPER_CLONE_DIR=%WHISPER_CLONE_DIR%
echo AUDACITY_CLONE_DIR=%AUDACITY_CLONE_DIR%
echo BUILD_FOLDER=%BUILD_FOLDER%

:: Save env. var's to a .bat to then source in next build steps.
echo set AUDACITY_BUILD_LEVEL=%AUDACITY_BUILD_LEVEL% > env.bat
echo set AUDACITY_BUILD_CONFIG=%AUDACITY_BUILD_CONFIG%>> env.bat
echo set AI_PLUGIN_VERSION=%AI_PLUGIN_VERSION%>> env.bat
echo set AI_PLUGIN_REPO_SOURCE_FOLDER=%AI_PLUGIN_REPO_SOURCE_FOLDER%>> env.bat
echo set LIBTORCH_DIR=%LIBTORCH_DIR%>> env.bat
echo set OPENVINO_DIR=%OPENVINO_DIR%>> env.bat
echo set OPENCL_SDK_DIR=%OPENCL_SDK_DIR%>> env.bat
echo set WHISPER_CLONE_DIR=%WHISPER_CLONE_DIR%>> env.bat
echo set AUDACITY_CLONE_DIR=%AUDACITY_CLONE_DIR%>> env.bat

set BUILD_FOLDER=%cd%
echo set BUILD_FOLDER=%BUILD_FOLDER%>> env.bat

if "%~6"=="" (
    goto end
)

call :ConvertRelToAbsPath %6 || exit /b 1
set CONAN_CACHE_ABS_PATH=%ABS_PATH%
set CONAN_HOME=%CONAN_CACHE_ABS_PATH%

echo CONAN_HOME=%CONAN_HOME%
echo set CONAN_HOME=%CONAN_HOME%>> env.bat

goto end

:: Helper function to check existance of a path, and convert it to an absolute path
:ConvertRelToAbsPath
set ABS_PATH=
set "rel_path=%~1"
if not exist "%rel_path%" (
    echo %rel_path% does not exist.
    exit /b 1
)
for %%i in ("%rel_path%") do set "ABS_PATH=%%~fi"

goto :eof

:error
echo Error: First 5 arguments are required, last 1 -- the conan_cache_path is optional. 
echo Usage: set_env.bat libtorch_location openvino_location opencl_sdk_location whisper_clone_location audacity_clone_location [conan_cache_path]
exit /b 1

:end
