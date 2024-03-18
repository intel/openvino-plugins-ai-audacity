# Audacity OpenVINO module build for Linux (Ubuntu 22.04)

Hi! The following is the process that we use when building the Audacity modules for Linux. These instructions are for Ubuntu 22.04, so you may need to adapt them slightly for other Linux distros.

## High-Level Overview
Before I get into the specifics, at a high-level we will be doing the following:
* Cloning & building whisper.cpp with OpenVINO support (For transcription audacity module)
* Cloning & building Audacity without modifications (just to make sure 'vanilla' build works fine)
* Adding our OpenVINO module src's to the Audacity source tree, and re-building it.

## Dependencies
Here are some of the dependencies that you need to grab. If applicable, I'll also give the cmd's to set up your environment here.
* Build Essentials (GCC & CMake)
```
sudo apt install build-essential
```
* OpenVINO - Download appropriate version from [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.0/linux). For these instructions, we will use ```l_openvino_toolkit_ubuntu22_*x86_64tgz```.
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
* OpenVINO Tokenizers Extension - Download package from [here](https://storage.openvinotoolkit.org/repositories/openvino_tokenizers/packages/2024.0.0.0/). For these instructions, we will use ```openvino_tokenizers_ubuntu22_2024.0.0.0_x86_64.tgz```.
```
# extract it 
tar xzvf openvino_tokenizers_ubuntu22_2024.0.0.0_x86_64.tgz

# copy extension libraries into OpenVINO lib folder:
cp openvino_tokenizers_ubuntu22_2024.0.0.0_x86_64/* l_openvino_toolkit_ubuntu22_2024.0.0.14509.34caeefd078_x86_64/runtime/lib/intel64/
```

* Libtorch (C++ distribution of pytorch)- This is a dependency for many of the pipelines that we ported from pytorch (musicgen, htdemucs, etc). We are currently using this version: [libtorch-cxx11-abi-shared-with-deps-2.1.1+cpu.zip ](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.1%2Bcpu.zip). Setup environment like this:
```
unzip libtorch-cxx11-abi-shared-with-deps-2.1.1+cpu.zip
export LIBTORCH_ROOTDIR=/path/to/libtorch
```

## Sub-Component builds
We're now going to build whisper.cpp.  
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

# Optional: You may want to check out specific tag here, such as Audacity-3.4.2
# If so, replace <Audacity_Tag> in following command with tag.
cd audacity
git checkout <Audacity_Tag>
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

# Optional: You may want to check out specific tag here, such as v3.4.2-R1
# If so, replace <Audacity_AI_Tag> in the following command.
cd openvino-plugins-ai-audacity
git checkout <Audacity_AI_Tag>
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
And we're done, at least with the module build. To actually use these modules, we need to generate / populate ```/usr/local/lib/``` with the OpenVINO models that the plugins will look for. At runtime, the plugins will look for these models in a openvino-models directory.
Here are the commands that you can use to create this directory, and populate it with the required models.

```
# Create an empty 'openvino-models' directory to start with
mkdir openvino-models

# Since many of these models will come from huggingdface repos, let's make sure git lfs is installed
sudo apt install git-lfs

#************
#* MusicGen *
#************
mkdir openvino-models/musicgen

# clone the HF repo
git clone https://huggingface.co/Intel/musicgen-static-openvino

# unzip the 'base' set of models (like the EnCodec, tokenizer, etc.) into musicgen folder
unzip musicgen-static-openvino/musicgen_small_enc_dec_tok_openvino_models.zip -d openvino-models/musicgen

# unzip the mono-specific set of models
unzip musicgen-static-openvino/musicgen_small_mono_openvino_models.zip -d openvino-models/musicgen

# unzip the stereo-specific set of models
unzip musicgen-static-openvino/musicgen_small_stereo_openvino_models.zip -d openvino-models/musicgen

# Now that the required models are extracted, feel free to delete the cloned 'musicgen-static-openvino' directory.
rm -rf musicgen-static-openvino

#*************************
#* Whisper Transcription *
#*************************

# clone the HF repo
git clone https://huggingface.co/Intel/whisper.cpp-openvino-models

# Extract the individual model packages into openvino-models directory
unzip whisper.cpp-openvino-models/ggml-base-models.zip -d openvino-models
unzip whisper.cpp-openvino-models/ggml-small-models.zip -d openvino-models
unzip whisper.cpp-openvino-models/ggml-small.en-tdrz-models.zip -d openvino-models

# Now that the required models are extracted, feel free to delete the cloned 'whisper.cpp-openvino-models' directory.
rm -rf whisper.cpp-openvino-models

#********************
#* Music Separation *
#********************

# clone the HF repo
git clone https://huggingface.co/Intel/demucs-openvino

# Copy the demucs OpenVINO IR files
cp demucs-openvino/htdemucs_v4.bin openvino-models
cp demucs-openvino/htdemucs_v4.xml openvino-models

# Now that the required models are extracted, feel free to delete the cloned 'demucs-openvino' directory.
rm -rf demucs-openvino

#*********************
#* Noise Suppression *
#*********************

# Clone the deepfilternet HF repo
git clone https://huggingface.co/Intel/deepfilternet-openvino

# extract deepfilter2 models
unzip deepfilternet-openvino/deepfilternet2.zip -d openvino-models

# extract deepfilter3 models
unzip deepfilternet-openvino/deepfilternet3.zip -d openvino-models

# For noise-suppression-denseunet-ll-0001, we can wget IR from openvino repo
cd openvino-models
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16/noise-suppression-denseunet-ll-0001.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16/noise-suppression-denseunet-ll-0001.bin
cd ..
```
After the above sequence of commands you should have a single ```openvino-models``` folder, which you can copy to /usr/local/lib like this:
```
sudo cp -R openvino-models /usr/local/lib/
```

# Need Help? :raising_hand_man:
For any questions about this build procedure, feel free to submit an issue [here](https://github.com/intel/openvino-plugins-ai-audacity/issues)
