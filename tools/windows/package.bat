@echo off
setlocal

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

IF "%AUDACITY_BUILD_CONFIG%"=="" (
    echo AUDACITY_BUILD_CONFIG is not set. Exiting.
    exit /b
)

IF "%AI_PLUGIN_VERSION%"=="" (
    echo AI_PLUGIN_VERSION is not set. Exiting.
    exit /b
)


set "bat_path=%~dp0"
set "audacity_ai_plugins_iss_path=%bat_path%audacity_ai_plugins.iss


iscc /O+ %audacity_ai_plugins_iss_path% ^
  /DBUILD_FOLDER=%BUILD_FOLDER% ^
  /DOPENVINO_DIR=%OPENVINO_DIR% ^
  /DOPENVINO_TOKENIZERS_DIR=%OPENVINO_TOKENIZERS_DIR% ^
  /DLIBTORCH_DIR=%LIBTORCH_DIR% ^
  /DAUDACITY_BUILD_CONFIG=%AUDACITY_BUILD_CONFIG% ^
  /DAI_PLUGIN_VERSION=%AI_PLUGIN_VERSION% ^
  /O%BUILD_FOLDER% ^
  /Faudacity-win-%AI_PLUGIN_VERSION%-64bit-OpenVINO-AI-Plugins
  

endlocal