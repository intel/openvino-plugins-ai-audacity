@echo off
setlocal 

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

echo Starting PreReq Routine...
set "prereq_bat=%bat_path%prereq.bat
call %prereq_bat%

echo Starting Build Routine...
set "build_bat=%bat_path%build.bat
call %build_bat%

echo Build complete. Packaging...
set "package_bat=%bat_path%package.bat
call %package_bat%

echo done...
exit /b
goto :eof

