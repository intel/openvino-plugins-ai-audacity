@echo off
setlocal enabledelayedexpansion

set "bat_path=%~dp0"

:::::::::::::::::::::::::::::::::
:: Package URL's and checksums ::
:::::::::::::::::::::::::::::::::
set LIBTORCH_PACKAGE_URL="https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.1%%%%2Bcpu.zip"
set LIBTORCH_PACKAGE_256SUM=e7b8d0b3b958d2215f52ff5385335f93aa78e42005727e44f1043d94d5bfc5dd

set OPENVINO_GENAI_PACKAGE_URL=https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2025.2.0.0rc3/openvino_genai_windows_2025.2.0.0rc3_x86_64.zip
set OPENVINO_GENAI_PACKAGE_256SUM=ac2cefec131d4faaaf217507693c5c1d5e8588bfcc3ac4ab56ea2b532ad80812

set OPENCL_SDK_URL=https://github.com/KhronosGroup/OpenCL-SDK/releases/download/v2023.04.17/OpenCL-SDK-v2023.04.17-Win-x64.zip
set OPENCL_SDK_256SUM=11844a1d69a71f82dc14ce66382c6b9fc8a4aee5840c21a786c5accb1d69bc0a

:::::::::::::::::::::::::::::
::  GIT Repo Configuration ::
:::::::::::::::::::::::::::::
set AUDACITY_REPO_CLONE_URL=https://github.com/audacity/audacity.git
set AUDACITY_REPO_CHECKOUT=release-3.7.4

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

call :DownloadVerifyExtract %OPENVINO_GENAI_PACKAGE_URL% %OPENVINO_GENAI_PACKAGE_256SUM%
IF "%EXTRACTED_PACKAGE_PATH%"=="" (
echo Error in openvino genai download routine..
exit /b
)

set OPENVINO_GENAI_DIR=%EXTRACTED_PACKAGE_PATH%

call :DownloadVerifyExtract %OPENCL_SDK_URL% %OPENCL_SDK_256SUM%
IF "%EXTRACTED_PACKAGE_PATH%"=="" (
echo Error in opencl tokenizers download routine..
exit /b
)

set OPENCL_SDK_DIR=%EXTRACTED_PACKAGE_PATH%

:: Clone the required repo's and check out the desired tags
git clone --depth 1 --branch %AUDACITY_REPO_CHECKOUT% %AUDACITY_REPO_CLONE_URL%

:: Create local python env, just to install conan.
python -m venv build_env

echo "activating..."
call "build_env\Scripts\activate"

echo "installing conan"
pip install conan

call %bat_path%\set_env.bat %LIBTORCH_DIR% %OPENVINO_GENAI_DIR% %OPENCL_SDK_DIR% audacity %CONAN_CACHE_PATH%

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
