# How to use build script :hammer:

## Dependencies
Here are some of the dependencies that you need to grab. If applicable, I'll also give the cmd's to set up your environment here.

* CMake (https://cmake.org/download/)
* Visual Studio (MS VS 2019 / 2022 Community Edition is fine)
* python3 / pip - Audacity requires conan 2.0+ to be installed, and the recommended way to do that is through pip.  
* Inno Setup -- This is used to package the build into an installer. You can download and install the latest stable Inno Setup installer here: https://jrsoftware.org/isdl.php#stable
  After installing, I would recommend explicitly adding the Inno Setup installation folder to your Path. For example:
  ```
  set Path="C:\Program Files (x86)\Inno Setup 6";%Path%
  ```

  ## How to run.
  If you have the above dependencies installed, you should be able to open a cmd.exe shell and do:
  ```
  fetch_build_package.bat
  ```

  This batch script will:
  1. Fetch required build dependencies like OpenVINO, LibTorch, OpenCL, etc.
  2. Build everything
  3. Package it all up into an installer using Inno Setup.

  One of the first thing thats that the script will do is create a 'BuildArtifacts' folder, with timestamp appended. For example:  
   ```BuildArtifacts-20240327_101701```
  
  All of the build collateral, including the downloaded packages and final output installer will be saved to this directory.
