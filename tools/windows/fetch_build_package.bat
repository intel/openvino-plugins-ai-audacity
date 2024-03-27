@echo off
setlocal 

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: The following set of variables are most likely what you might want to modify.
:: Details about the version of Audacity that we will build against.
set AUDACITY_REPO_CHECKOUT=Audacity-3.4.2
set AUDACITY_BUILD_LEVEL=2
set AUDACITY_BUILD_CONFIG=RelWithDebInfo

:: The version that we will pass to inno setup as the app version.
set AI_PLUGIN_VERSION=v3.4.2-R2

:: The branch or tag of whisper.cpp that we will build against.
set WHISPERCPP_REPO_CHECKOUT=v1.5.4

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Create Build Folder
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
set "bat_path=%~dp0"

:: Get date and time in YYYYMMDD_HHMMSS format
set "datestamp=%DATE:~-4%%DATE:~4,2%%DATE:~7,2%"
set "timestamp=%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "timestamp=%timestamp: =0%"

:: Combine them to form a folder name
set "BUILD_FOLDER=%cd%\BuildArtifacts-%datestamp%_%timestamp%"

echo BUILD_FOLDER=%BUILD_FOLDER%

:: Create the folder
md "%BUILD_FOLDER%"

:: Go into the build folder
cd "%BUILD_FOLDER%"

:::::::::::::::::::::::::::
:: Download the packages ::
set EXTRACTED_PACKAGE_PATH=

set LIBTORCH_PACKAGE_URL="https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.1%%%%2Bcpu.zip"
set LIBTORCH_PACKAGE_256SUM=0ee1879b5d864a18eff555dd4f42051addb257a40abe347f4691f2bc4c039293

call :DownloadVerifyExtract %LIBTORCH_PACKAGE_URL% %LIBTORCH_PACKAGE_256SUM% libtorch.zip

IF "%EXTRACTED_PACKAGE_PATH%"=="" (
echo Error in libtorch download routine..
exit /b
)

set LIBTORCH_DIR=%EXTRACTED_PACKAGE_PATH%

set OPENVINO_PACKAGE_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.0/windows/w_openvino_toolkit_windows_2024.0.0.14509.34caeefd078_x86_64.zip
set OPENVINO_PACKAGE_256SUM=764ba560fc79de67a7e3f183a15eceb97eeda9a60032e3dd6866f7996745ed9d

call :DownloadVerifyExtract %OPENVINO_PACKAGE_URL% %OPENVINO_PACKAGE_256SUM%

IF "%EXTRACTED_PACKAGE_PATH%"=="" (
echo Error in openvino download routine..
exit /b
)

set OPENVINO_DIR=%EXTRACTED_PACKAGE_PATH%

set OPENVINO_TOKENIZERS_URL=https://storage.openvinotoolkit.org/repositories/openvino_tokenizers/packages/2024.0.0.0/openvino_tokenizers_windows_2024.0.0.0_x86_64.zip
set OPENVINO_TOKENIZERS_256SUM=ca2e5e893fd2cebfe8f58542d00804524b80c60d9c23bd0424c55a996222bad7

call :DownloadVerifyExtract %OPENVINO_TOKENIZERS_URL% %OPENVINO_TOKENIZERS_256SUM%


IF "%EXTRACTED_PACKAGE_PATH%"=="" (
echo Error in openvino tokenizers download routine..
exit /b
)

set OPENVINO_TOKENIZERS_DIR=%EXTRACTED_PACKAGE_PATH%

echo EXTRACTED_PACKAGE_PATH=%EXTRACTED_PACKAGE_PATH%

set OPENCL_SDK_URL=https://github.com/KhronosGroup/OpenCL-SDK/releases/download/v2023.04.17/OpenCL-SDK-v2023.04.17-Win-x64.zip
set OPENCL_SDK_256SUM=11844a1d69a71f82dc14ce66382c6b9fc8a4aee5840c21a786c5accb1d69bc0a

call :DownloadVerifyExtract %OPENCL_SDK_URL% %OPENCL_SDK_256SUM%

IF "%EXTRACTED_PACKAGE_PATH%"=="" (
echo Error in opencl tokenizers download routine..
exit /b
)

set OPENCL_SDK_DIR=%EXTRACTED_PACKAGE_PATH%

set AI_PLUGIN_REPO_SOURCE_FOLDER=%bat_path%\..\..\

echo AI_PLUGIN_REPO_SOURCE_FOLDER=%AI_PLUGIN_REPO_SOURCE_FOLDER%

echo Done fetching packages.
echo OPENVINO_DIR=%OPENVINO_DIR%
echo OPENVINO_TOKENIZERS_DIR=%OPENVINO_TOKENIZERS_DIR%
echo OPENCL_SDK_DIR=%OPENCL_SDK_DIR%
echo LIBTORCH_DIR=%LIBTORCH_DIR%

echo Starting Build Routine...
set "build_bat=%bat_path%build.bat
call %build_bat%

echo Build complete. Packaging...
set "package_bat=%bat_path%package.bat
call %package_bat%

echo done...
exit /b
goto :eof

:: Helper function to Download, verify it using checksum, and then extract.
:DownloadVerifyExtract
set EXTRACTED_PACKAGE_PATH=
set "package_url=%~1"
set "checksum=%~2"
set "package_file=%~3"

IF "%package_file%"=="" (
:: package_file isn't explicitly set. 
for %%i in (%package_url%) do set "package_file=%%~nxi"
)

echo package_file=%package_file%

echo Downloading from %package_url%
powershell -Command "Invoke-WebRequest -Uri '%package_url%' -OutFile %package_file%" 

CertUtil -hashfile %package_file% SHA256 | find /i /v "sha256" | find /i /v "certutil" >computed_checksum.txt
set /p COMPUTED_CHECKSUM=<computed_checksum.txt

if "%checksum%" == "%COMPUTED_CHECKSUM%" (
    echo Checksum verification successful.
) else (
    echo Checksum verification failed.
	goto :eof
)

echo Extracting %package_file% to %BUILD_FOLDER% ...
powershell -Command "Expand-Archive -LiteralPath '%package_file%' -DestinationPath '%BUILD_FOLDER%' -Force"

for %%A in ("%package_file%") do set "package_folder=%%~nA"

set EXTRACTED_PACKAGE_PATH=%BUILD_FOLDER%\%package_folder%
goto :eof


endlocal
