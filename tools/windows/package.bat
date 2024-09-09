@echo off
setlocal

IF NOT EXIST env.bat (
    echo env.bat is not set. Exiting.
    exit /b 1
)

call env.bat

IF "%BUILD_FOLDER%"=="" (
    echo BUILD_FOLDER is not set. Exiting.
    exit /b
)

IF "%OPENVINO_DIR%"=="" (
    echo OPENVINO_DIR is not set. Exiting.
    exit /b
)

IF "%LIBTORCH_DIR%"=="" (
    echo LIBTORCH_DIR is not set. Exiting.
    exit /b
)

IF "%AUDACITY_BUILD_CONFIG%"=="" (
    echo AUDACITY_BUILD_CONFIG is not set. Exiting.
    exit /b
)

IF "%AI_PLUGIN_VERSION%"=="" (
    echo AI_PLUGIN_VERSION is not set. Exiting.
    exit /b
)

IF "%AI_PLUGIN_REPO_SOURCE_FOLDER%"=="" (
    echo AI_PLUGIN_REPO_SOURCE_FOLDER is not set. Exiting.
    exit /b
)

IF "%OPENCL_SDK_DIR%"=="" (
    echo OPENCL_SDK_DIR is not set. Exiting.
    exit /b
)



set "bat_path=%~dp0"
set "audacity_ai_plugins_iss_path=%bat_path%audacity_ai_plugins.iss


iscc /O+ %audacity_ai_plugins_iss_path% ^
  /DBUILD_FOLDER=%BUILD_FOLDER% ^
  /DOPENVINO_DIR=%OPENVINO_DIR% ^
  /DLIBTORCH_DIR=%LIBTORCH_DIR% ^
  /DAUDACITY_BUILD_CONFIG=%AUDACITY_BUILD_CONFIG% ^
  /DAI_PLUGIN_VERSION=%AI_PLUGIN_VERSION% ^
  /DAI_PLUGIN_REPO_SOURCE_FOLDER=%AI_PLUGIN_REPO_SOURCE_FOLDER% ^
  /DOPENCL_SDK_DIR=%OPENCL_SDK_DIR% ^
  /O%BUILD_FOLDER% ^
  /Faudacity-win-%AI_PLUGIN_VERSION%-64bit-OpenVINO-AI-Plugins

:: Get date and time in YYYYMMDD_HHMMSS format
set "datestamp=%DATE:~-4%%DATE:~4,2%%DATE:~7,2%"
set "timestamp=%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "timestamp=%timestamp: =0%"

rename audacity-win-%AI_PLUGIN_VERSION%-64bit-OpenVINO-AI-Plugins.exe audacity-win-%AI_PLUGIN_VERSION%-64bit-OpenVINO-AI-Plugins-%datestamp%_%timestamp%.exe

echo Done! Generated %cd%\audacity-win-%AI_PLUGIN_VERSION%-64bit-OpenVINO-AI-Plugins-%datestamp%_%timestamp%.exe

endlocal