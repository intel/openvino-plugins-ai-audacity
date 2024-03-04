# Audacity OpenVINO module build for Linux (Ubuntu 22.04)

Hi! The following is the process that we use when building the Audacity modules for Linux. These instructions are for Ubuntu 22.04, so you may need to adapt them slightly for other Linux distros.

## High-Level Overview
Before I get into the specifics, at a high-level we will be doing the following:
* Cloning & building whisper.cpp with OpenVINO support (For transcription audacity module)
* Cloning & building openvino-stable-diffusion-cpp (This is to support Music Generation & Remix features)
* Cloning & building Audacity without modifications (just to make sure 'vanilla' build works fine)
* Adding our OpenVINO module src's to the Audacity source tree, and re-building it.

## Dependencies
Here are some of the dependencies that you need to grab. If applicable, I'll also give the cmd's to set up your environment here.
* Build Essentials (GCC & CMake)
```
sudo apt install build-essential
```
* OpenVINO - Download appropriate version from [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/). For these instructions, we will use ```l_openvino_toolkit_ubuntu22_*x86_64tgz```.
```
# Extract it
tar xvf l_openvino_toolkit_ubuntu22_*x86_64.tgz 

#install dependencies
cd l_openvino_toolkit_ubuntu22_*_x86_64/install_dependencies/
sudo -E ./install_openvino_dependencies.sh
cd ..

# setup env
source setupvars.sh
```
* OpenVINO Tokenizers Extension - Download package from [here](https://storage.openvinotoolkit.org/repositories/openvino_tokenizers/packages/2023.3.0.0/). For these instructions, we will use ```openvino_tokenizers_ubuntu22_2023.3.0.0_x86_64.tgz```.
```
# extract it 
tar xzvf openvino_tokenizers_ubuntu22_2023.3.0.0_x86_64.tgz

# copy extension libraries into OpenVINO lib folder:
cp openvino_tokenizers_ubuntu22_2023.3.0.0_x86_64/* l_openvino_toolkit_ubuntu22_2023.3.0.13775.ceeafaf64f3_x86_64/runtime/lib/intel64/
```

* OpenCV - Only a dependency for the  OpenVINO Stable-Diffusion CPP samples (to read/write images from disk, display images, etc.). You can install like this:
```
sudo apt install libopencv-dev
```
* Libtorch (C++ distribution of pytorch)- This is a dependency for the audio utilities in openvino-stable-diffusion-cpp (like spectrogram-to-wav, wav-to-spectrogram), as well as some of our htdemucs v4 routines (supporting music separation). We are currently using this version: [libtorch-cxx11-abi-shared-with-deps-2.1.1+cpu.zip ](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.1%2Bcpu.zip). Setup environment like this:
```
unzip libtorch-cxx11-abi-shared-with-deps-2.1.1+cpu.zip
export LIBTORCH_ROOTDIR=/path/to/libtorch
```

## Sub-Component builds
We're now going to build whisper.cpp, stablediffusion-pipelines-cpp, and sentencepiece.  
```
# OpenVINO
source /path/to/l_openvino_toolkit_ubuntu22_*_x86_64/setupvars.sh

# Libtorch
export LIBTORCH_ROOTDIR=/path/to/libtorch
```

### Whisper.cpp 
```
# Clone it & check out specific tag
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
git checkout v1.5.4
cd ..

# Create build folder
mkdir whisper-build
cd whisper-build

# Run CMake, specifying that you want to enable OpenVINO support.
cmake ../whisper.cpp/ -DWHISPER_OPENVINO=ON

# Build it:
make -j 4

# Install built whisper collateral into a local 'installed' directory:
cmake --install . --config Release --prefix ./installed
```
With the build / install complete, the Audacity build will find the built collateral via the WHISPERCPP_ROOTDIR. So you can set it like this:
```
export WHISPERCPP_ROOTDIR=/path/to/whisper-build/installed
export LD_LIBRARY_PATH=${WHISPERCPP_ROOTDIR}/lib:$LD_LIBRARY_PATH
```
(I'll remind you later about this though)

### OpenVINO Stable-Diffusion CPP
```
# clone it & check out v0.1 tag
git clone https://github.com/intel/stablediffusion-pipelines-cpp.git
cd stablediffusion-pipelines-cpp
git checkout v0.1
cd ..

#create build folder
mkdir stablediffusion-pipelines-cpp-build
cd stablediffusion-pipelines-cpp-build

# Run cmake
cmake ../stablediffusion-pipelines-cpp

# Build it
make -j 8

# Install it
cmake --install . --config Release --prefix ./installed

# Set environment variable that Audacity module will use to find this component.
export CPP_STABLE_DIFFUSION_OV_ROOTDIR=/path/to/stablediffusion-pipelines-cpp-build/installed
export LD_LIBRARY_PATH=${CPP_STABLE_DIFFUSION_OV_ROOTDIR}/lib:$LD_LIBRARY_PATH

```

## Audacity 

Okay, moving on to actually building Audacity. Just a reminder, we're first going to just build Audacity without any modifications. Once that is done, we'll copy our openvino-module into the Audacity src tree, and built that.

### Audacity initial (vanilla) build
```
#Install some build dependencies
sudo apt-get install -y build-essential cmake git python3-pip
sudo pip3 install conan
sudo apt-get install libgtk2.0-dev libasound2-dev libjack-jackd2-dev uuid-dev

# clone Audacity
git clone https://github.com/audacity/audacity.git

# Check out Audacity-3.4.2 tag, 
cd audacity
git checkout Audacity-3.4.2
cd ..

# Create build directory
mkdir audacity-build
cd audacity-build

# Run cmake (grab a coffee & a snack... this takes a while)
cmake -G "Unix Makefiles" ../audacity -DCMAKE_BUILD_TYPE=Release

# build it 
make -j`nproc`
```

When this is done, you can run Audacity like this (from audacity-build directory):
```
Release/bin/audacity
```

### Audacity OpenVINO module build

Now we'll run through the steps to actually build the OpenVINO-based Audacity module.

First, clone the following repo. This is where the actual Audacity module code lives today.
```
:: clone it
git clone https://github.com/intel/openvino-plugins-ai-audacity.git

# Check out the release tag that matches the Audacity version you're using
cd openvino-plugins-ai-audacity
git checkout v3.4.2-R1
cd ..
```

We need to copy the ```mod-openvino``` folder into the Audacity source tree.
i.e. Copy ```openvino-plugins-ai-audacity/mod-openvino``` folder to ```audacity/modules```.


We now need to edit ```audacity\modules\CMakeLists.txt``` to add mod-openvino as a build target. You just need to add a ```add_subdirectory(mod-openvino)``` someplace in the file. For example:

```
...
foreach( MODULE ${MODULES} )
   add_subdirectory("${MODULE}")
endforeach()

#YOU CAN ADD IT HERE
add_subdirectory(mod-openvino)

if( NOT CMAKE_SYSTEM_NAME MATCHES "Darwin" )
   if( NOT "${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
      install( DIRECTORY "${_DEST}/modules"
               DESTINATION "${_PKGLIB}" )
   endif()
endif()
...
```

Okay, now we're going to (finally) build the module. Here's a recap of the environment variables that you should have set:

```
# OpenVINO
source /path/to/l_openvino_toolkit_ubuntu22_*_x86_64/setupvars.sh

# Libtorch
export LIBTORCH_ROOTDIR=/path/to/libtorch

# Whisper.cpp
export WHISPERCPP_ROOTDIR=/path/to/whisper-build/installed
export LD_LIBRARY_PATH=${WHISPERCPP_ROOTDIR}/lib:$LD_LIBRARY_PATH

# CPP Stable Diffusion
export CPP_STABLE_DIFFUSION_OV_ROOTDIR=/path/to/stablediffusion-pipelines-cpp-build/installed
export LD_LIBRARY_PATH=${CPP_STABLE_DIFFUSION_OV_ROOTDIR}/lib:$LD_LIBRARY_PATH
```

Okay, on to the build:  
```
# cd back to the same Audacity folder you used to build Audacity before
cd audacity-build

# and re-run cmake step (it will go faster this time)
cmake -G "Unix Makefiles" ../audacity -DCMAKE_BUILD_TYPE=Release

# and re-run make command
make -j`nproc`
```

If it all builds correctly, you should see mod-openvino.so sitting in Release/lib/audacity/modules/

You can go ahead and run audacity-build/Release/bin/audacity

Once Audacity is open, you need to go to Edit->Preferences. And on the left side you'll see a 'Modules' tab, click that. And here you (hopefully) see mod-openvino entry set to 'New'. You need to change it to 'Enabled', as shown in the following picture.  

![](preferences_enabled.png)

Once you change to 'Enabled', close Audacity and re-open it. When it comes back up, you should now see the OpenVINO modules listed.

## OpenVINO Models Installation  
And we're done, at least with the module build. To actually use these modules, we need to generate / populate ```/usr/local/lib/``` with the models for noise-separation, music separation (htdemucs), whisper, and riffusion. Start by downloading the model package zip file from here:  
[openvino-models.zip](https://github.com/intel/openvino-plugins-ai-audacity/releases/download/v3.4.2-R1/openvino-models.zip)

And extract / install them into /usr/local/lib like this:
```
# unzip the packages. 
unzip openvino-models.zip

# After above command you should have a single ```openvino-models``` folder, which you can copy to /usr/local/lib:
sudo cp -R openvino-models /usr/local/lib/
```

# Need Help? :raising_hand_man:
For any questions about this build procedure, feel free to submit an issue [here](https://github.com/intel/openvino-plugins-ai-audacity/issues)
