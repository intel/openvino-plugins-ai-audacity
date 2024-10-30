@echo off
setlocal enabledelayedexpansion

set "bat_path=%~dp0"

:::::::::::::::::::::::::::::::::
:: Package URL's and checksums ::
:::::::::::::::::::::::::::::::::
set LIBTORCH_PACKAGE_URL="https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.1%%%%2Bcpu.zip"
set LIBTORCH_PACKAGE_256SUM=e7b8d0b3b958d2215f52ff5385335f93aa78e42005727e44f1043d94d5bfc5dd

set OPENVINO_PACKAGE_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.3/windows/w_openvino_toolkit_windows_2024.3.0.16041.1e3b88e4e3f_x86_64.zip
set OPENVINO_PACKAGE_256SUM=bd2ea0d5dc7936f375f47dfeb5ddaa59b42f392057a686f61672f1d258394c6c

set OPENVINO_TOKENIZERS_URL=https://storage.openvinotoolkit.org/repositories/openvino_tokenizers/packages/2024.3.0.0/openvino_tokenizers_windows_2024.3.0.0_x86_64.zip
set OPENVINO_TOKENIZERS_256SUM=4e0b33e31c182204bdefd1d22fcde111836fc82fe31d72cd48de1c3981ce8410

set OPENCL_SDK_URL=https://github.com/KhronosGroup/OpenCL-SDK/releases/download/v2023.04.17/OpenCL-SDK-v2023.04.17-Win-x64.zip
set OPENCL_SDK_256SUM=11844a1d69a71f82dc14ce66382c6b9fc8a4aee5840c21a786c5accb1d69bc0a

:::::::::::::::::::::::::::::
::  GIT Repo Configuration ::
:::::::::::::::::::::::::::::
set AUDACITY_REPO_CLONE_URL=https://github.com/audacity/audacity.git
set AUDACITY_REPO_CHECKOUT=release-3.7.0

set WHISPERCPP_REPO_CLONE_URL=https://github.com/ggerganov/whisper.cpp
set WHISPERCPP_REPO_CHECKOUT=v1.6.0

::::::::::::::::::::::::::::::::::::::::::::::::
:: Download, verify, and extract the packages ::
::::::::::::::::::::::::::::::::::::::::::::::::

if not defined LIBTORCH_DIR (
	call :DownloadVerifyExtract %LIBTORCH_PACKAGE_URL% %LIBTORCH_PACKAGE_256SUM% libtorch.zip
	IF "!EXTRACTED_PACKAGE_PATH!"=="" (
	echo Error in libtorch download routine..
	exit /b
	)
	set LIBTORCH_DIR=!EXTRACTED_PACKAGE_PATH!
) else (
    echo Not downloading Libtorch, as LIBTORCH_DIR is defined by environment. LIBTORCH_DIR=%LIBTORCH_DIR%
)

call :DownloadVerifyExtract %OPENVINO_PACKAGE_URL% %OPENVINO_PACKAGE_256SUM%
IF "%EXTRACTED_PACKAGE_PATH%"=="" (
echo Error in openvino download routine..
exit /b
)

set OPENVINO_DIR=%EXTRACTED_PACKAGE_PATH%

call :DownloadVerifyExtract %OPENVINO_TOKENIZERS_URL% %OPENVINO_TOKENIZERS_256SUM% openvino_tokenizers.zip
IF "%EXTRACTED_PACKAGE_PATH%"=="" (
echo Error in openvino tokenizers download routine..
exit /b
)

echo Extracting openvino_tokenizers.zip to %OPENVINO_DIR% ...
powershell -Command "Expand-Archive -LiteralPath openvino_tokenizers.zip -DestinationPath '%OPENVINO_DIR%' -Force"

call :DownloadVerifyExtract %OPENCL_SDK_URL% %OPENCL_SDK_256SUM%
IF "%EXTRACTED_PACKAGE_PATH%"=="" (
echo Error in opencl tokenizers download routine..
exit /b
)

set OPENCL_SDK_DIR=%EXTRACTED_PACKAGE_PATH%

:: Clone the required repo's and check out the desired tags
git clone --depth 1 --branch %WHISPERCPP_REPO_CHECKOUT% %WHISPERCPP_REPO_CLONE_URL%

git clone --depth 1 --branch %AUDACITY_REPO_CHECKOUT% %AUDACITY_REPO_CLONE_URL%

:: Create local python env, just to install conan.
python -m venv build_env

echo "activating..."
call "build_env\Scripts\activate"

echo "installing conan"
pip install conan

call %bat_path%\set_env.bat %LIBTORCH_DIR% %OPENVINO_DIR% %OPENCL_SDK_DIR% whisper.cpp audacity %CONAN_CACHE_PATH%

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
