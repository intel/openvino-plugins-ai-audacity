# Audacity OpenVINO Module – Build Guide for Arch Linux

> This guide is an Arch Linux adaptation of the official [Ubuntu 22.04 build instructions](https://github.com/intel/openvino-plugins-ai-audacity/blob/main/doc/build_doc/linux/README.md).
> All steps were adjusted to use `pacman` / AUR instead of `apt-get`, and to build OpenVINO from source, since no official pre-compiled OpenVINO drops exist for Arch Linux.
> Warning: This is not a permanent solution. It is the only way at the moment. I am working to get automatic updates and a release as an official AUR-package. If you want to help, please reach out to me.
> You will have to compile your own versions of Audacity, OpenVINO, e.g. There are no automatic updates.
> Classical problem: These steps work on my machine. I really do hope, it will also work on yours.
> During this process, you will build your own version of Audacity. You don't need it preinstalled on your computer. If you have it installed, you can either keep it (you will have 2 versions then) or simply uninstall it.

Throughout this guide, `~/audacity-openvino/` is used as the working directory. Feel free to adjust this to your liking.

---

## Table of Contents

1. [Install System Dependencies](#1-install-system-dependencies)
2. [Build OpenVINO from Source](#2-build-openvino-from-source)
3. [Set Up OpenVINO Tokenizers](#3-set-up-openvino-tokenizers)
4. [Download Libtorch](#4-download-libtorch)
5. [Build whisper.cpp with OpenVINO Support](#5-build-whispercpp-with-openvino-support)
6. [Build Audacity (Vanilla)](#6-build-audacity-vanilla)
7. [Add the OpenVINO Module and Rebuild](#7-add-the-openvino-module-and-rebuild)
8. [Enable the Module in Audacity](#8-enable-the-module-in-audacity)
9. [Download and Install AI Models](#9-download-and-install-ai-models)

---

## 1. Install System Dependencies

Unlike Ubuntu, Arch Linux uses `pacman` (and `yay`/`paru` for AUR packages).
The equivalent of the Ubuntu build dependencies is:

```bash
sudo pacman -S --needed \
  base-devel cmake git python python-pip \
  gtk3 alsa-lib \
  wget curl unzip

# Conan is the build tool required by Audacity 3.4+
python -m pip install conan --break-system-packages
```

> **Note:** `libuuid` is part of `util-linux`, which is installed by default on Arch – no need to install it separately.
> Audacity uses GTK3 (not GTK2). `libasound2-dev` maps to `alsa-lib`.
> Use `python -m pip` instead of `pip` directly, as the `pip` command may not be in PATH on a fresh Arch install.
> **Do NOT install `jack2`** – modern Arch systems use `pipewire-jack` as a drop-in JACK replacement. Installing `jack2` would conflict with and remove `pipewire-jack`, potentially breaking your audio setup. `pipewire-jack` is fully sufficient for the Audacity build.

---

## 2. Build OpenVINO from Source

Since Arch Linux has no official pre-compiled OpenVINO toolkit (unlike Ubuntu), we need to build it from source.
The AUR package `openvino` exists but has had stability issues — building manually from source is more reliable.

```bash
mkdir -p ~/audacity-openvino
cd ~/audacity-openvino

# Install additional build dependencies for OpenVINO
sudo pacman -S --needed \
  python-setuptools python-wheel \
  tbb ocl-icd opencl-headers opencl-clhpp \
  pugixml snappy flatbuffers abseil-cpp

# IMPORTANT: If the `onnx` package is installed, remove it temporarily.
# ONNX stands for Open Neural Network Exchange. It is an open format built to represent machine learning models.
# Find out more about onnx: https://onnx.ai/
# Arch's onnx package has broken CMake target exports that conflict with OpenVINO.
# OpenVINO uses its own bundled onnx instead. You can reinstall onnx after the build.
sudo pacman -R onnx 2>/dev/null || true

# Clone OpenVINO
git clone --recurse-submodules https://github.com/openvinotoolkit/openvino.git
cd openvino
git checkout 2024.6.0
git submodule update --init --recursive
cd ..

# Create build directory
mkdir openvino-build
cd openvino-build

cmake ../openvino \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DENABLE_PYTHON=OFF \
  -DENABLE_TESTS=OFF \
  -DENABLE_SAMPLES=OFF \
  -DENABLE_DOCS=OFF \
  -DENABLE_INTEL_GPU=OFF \
  -DENABLE_INTEL_NPU=OFF \
  -DENABLE_CPPLINT=OFF

# Patch for GCC 15 compatibility: OpenVINO 2024.6.0 was written for GCC 14 and is missing a required #include <cstdint> in multiple files. GCC 15 (shipped with Arch) is stricter and requires it explicitly.
cd ~/audacity-openvino

sed -i '14a #include <cstdint>' \
  openvino/src/core/include/openvino/core/type/bfloat16.hpp

for f in \
  openvino/src/core/include/openvino/core/type/float16.hpp \
  openvino/src/core/include/openvino/core/type/float4_e2m1.hpp \
  openvino/src/core/include/openvino/core/type/float8_e4m3.hpp \
  openvino/src/core/include/openvino/core/type/float8_e5m2.hpp \
  openvino/src/core/include/openvino/core/type/float8_e8m0.hpp \
  openvino/src/core/dev_api/openvino/core/type/nf4.hpp; do
    sed -i '/#include "openvino\/core\/core_visibility.hpp"/a #include <cstdint>' "$f"
done

cd openvino-build
make -j$(nproc)

cmake --install . --prefix ~/audacity-openvino/openvino_toolkit/installed

cd ~/audacity-openvino
```

Set up the environment variables:

```bash
source ~/audacity-openvino/openvino_toolkit/installed/setupvars.sh
```

> **NOTE** Do NOT add this to your `~/bashrc`. It is only needed during the build process. Audacity will load the OpenVINO libraries automatically at runtime. Adding it to `.bashrc` will show a Python version waring on every terminal start. 

---

## 3. Set Up OpenVINO Tokenizers

```bash
cd ~/audacity-openvino

# Clone and build openvino_tokenizers
git clone https://github.com/openvinotoolkit/openvino_tokenizers.git
cd openvino_tokenizers
git checkout 2024.6.0

mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DOpenVINO_DIR=~/audacity-openvino/openvino_toolkit/installed/runtime/cmake

make -j$(nproc)

# If make fails with a 'uint32_t does not name a type' error in sentencepiece_processor.h,
# apply this GCC 15 patch and run make again:
#
#   sed -i '23a #include <cstdint>' \
#     ~/audacity-openvino/openvino_tokenizers/build/_deps/sentencepiece-src/src/sentencepiece_processor.h
#   make -j$(nproc)

cp ~/audacity-openvino/openvino_tokenizers/build/src/libopenvino_tokenizers.so \
   ~/audacity-openvino/openvino_toolkit/installed/runtime/lib/intel64/

cd ~/audacity-openvino
```

---

## 4. Download Libtorch

Libtorch is a C++ distribution of PyTorch. We use the CPU-only version:

```bash
cd ~/audacity-openvino

wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.1+cpu.zip

export LIBTORCH_ROOTDIR=~/audacity-openvino/libtorch
export LD_LIBRARY_PATH=${LIBTORCH_ROOTDIR}/lib:$LD_LIBRARY_PATH
```

---

## 5. Build whisper.cpp with OpenVINO Support

```bash
cd ~/audacity-openvino

git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
git checkout v1.5.4
cd ..

mkdir whisper-build
cd whisper-build

cmake ../whisper.cpp/ \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DWHISPER_OPENVINO=ON

make -j$(nproc)

cmake --install . --config Release --prefix ./installed

export WHISPERCPP_ROOTDIR=~/audacity-openvino/whisper-build/installed
export LD_LIBRARY_PATH=${WHISPERCPP_ROOTDIR}/lib:$LD_LIBRARY_PATH

cd ~/audacity-openvino
```

---

## 6. Build Audacity (Vanilla)

First, build a plain unmodified Audacity to confirm the build environment works.

> **Note on GCC versions:** Audacity's build system uses Conan to manage dependencies, and some of those dependencies (specifically `m4`) fail to compile with GCC 15 which ships with current Arch Linux. The solution is to install GCC 14 as an alternative compiler.
> 
> Unfortunately there is no precompiled `-bin` version of `gcc14` in the AUR — it must be built from source. **This can take 1–3 hours depending on your CPU.** Be prepared to wait. The build includes a full testsuite run which adds significant time — you can safely kill the test processes (look for `expect` processes via `ps aux | grep expect`) without affecting the installation.
> ```bash
> yay -S gcc14
> ```
> 
> After installation, verify gcc14 is correctly installed:
> ```bash
> gcc-14 --version
> # Expected output: gcc (GCC) 14.x.x ...
> ```

```bash

# gtk2 is required by Conan's wxWidgets build. It is no longer in the official Arch repos and must be installed from the AUR.
yay -S gtk2

git clone https://github.com/audacity/audacity.git
cd audacity
git checkout release-3.7.7
cd ..

mkdir audacity-build
cd audacity-build

CC=gcc-14 CXX=g++-14 cmake -G "Unix Makefiles" ../audacity \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

make -j$(nproc)
```

You should now be able to run Audacity:

```bash
~/audacity-openvino/audacity-build/Release/bin/audacity
```

---

## 7. Add the OpenVINO Module and Rebuild

```bash
cd ~/audacity-openvino

# Clone the OpenVINO plugin repo
git clone https://github.com/intel/openvino-plugins-ai-audacity.git

# Copy the module into the Audacity source tree
cp -r ~/audacity-openvino/openvino-plugins-ai-audacity/mod-openvino \
      ~/audacity-openvino/audacity/modules/

# Edit Audacity's module CMakeLists.txt to include mod-openvino
echo 'add_subdirectory(mod-openvino)' >> ~/audacity-openvino/audacity/modules/CMakeLists.txt
```

Now rebuild:

```bash
cd ~/audacity-openvino/audacity-build

CC=gcc-14 CXX=g++-14 cmake -G "Unix Makefiles" ../audacity \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DOpenVINO_DIR=~/audacity-openvino/openvino_toolkit/installed/runtime/cmake \
  -DWHISPERCPP_ROOTDIR=${WHISPERCPP_ROOTDIR} \
  -DLIBTORCH_ROOTDIR=${LIBTORCH_ROOTDIR}

make -j$(nproc)
```

If successful, you will find `mod-openvino.so` at:

```
~/audacity-openvino/audacity-build/Release/lib/audacity/modules/mod-openvino.so
```

> **⚠️ Important: Do NOT copy mod-openvino.so into your system-installed Audacity.**
> The self-compiled Audacity uses wxWidgets 3.2, while the Arch package uses wxWidgets 3.1.3. Copying the module will result in a fatal wxWidgets version mismatch error. Always use the self-compiled Audacity binary to run OpenVINO effects:
> ```bash
> ~/audacity-openvino/audacity-build/Release/bin/audacity
> ```

---

## 8. Enable the Module in Audacity

> **Note:** Always use the self-compiled Audacity binary, not the system-installed version. See the warning in Step 7.

1. Launch the self-compiled Audacity:
   ```bash
   ~/audacity-openvino/audacity-build/Release/bin/audacity
   ```
2. Go to **Edit → Preferences → Modules**
3. Find `mod-openvino` and set it to **Enabled**
4. Close and reopen Audacity
5. The OpenVINO AI effects should now appear in the **Effect** menu

---

## 9. Download and Install AI Models

The plugins require AI models to be present at `/usr/local/lib/openvino-models/`.

First install git-lfs:
```bash
sudo pacman -S git-lfs
git lfs install
```

Then create the models directory and download the models you need:
```bash
cd ~/audacity-openvino
mkdir openvino-models
```

### Music Separation (Demucs)
```bash
git clone --no-checkout https://huggingface.co/Intel/demucs-openvino
cd demucs-openvino
git checkout 97fc578fb57650045d40b00bc84c7d156be77547
cd ..

cp demucs-openvino/htdemucs_v4.bin openvino-models
cp demucs-openvino/htdemucs_v4.xml openvino-models
rm -rf demucs-openvino
```

### Noise Suppression (DeepFilterNet)
```bash
git clone --no-checkout https://huggingface.co/Intel/deepfilternet-openvino
cd deepfilternet-openvino
git checkout 995706bda3da69da0825074ba7dbc8a78067e980
cd ..

unzip deepfilternet-openvino/deepfilternet2.zip -d openvino-models
unzip deepfilternet-openvino/deepfilternet3.zip -d openvino-models
rm -rf deepfilternet-openvino

cd openvino-models
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16/noise-suppression-denseunet-ll-0001.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16/noise-suppression-denseunet-ll-0001.bin
cd ..
```

### Whisper Transcription
```bash
git clone https://huggingface.co/Intel/whisper.cpp-openvino-models

unzip whisper.cpp-openvino-models/ggml-base-models.zip -d openvino-models
unzip whisper.cpp-openvino-models/ggml-small-models.zip -d openvino-models
unzip whisper.cpp-openvino-models/ggml-small.en-tdrz-models.zip -d openvino-models
rm -rf whisper.cpp-openvino-models
```

### MusicGen
```bash
mkdir openvino-models/musicgen

git clone --no-checkout https://huggingface.co/Intel/musicgen-static-openvino
cd musicgen-static-openvino
git checkout b2ad8083f3924ed704814b68c5df9cbbf2ad2aae
cd ..

unzip musicgen-static-openvino/musicgen_small_enc_dec_tok_openvino_models.zip -d openvino-models/musicgen
unzip musicgen-static-openvino/musicgen_small_mono_openvino_models.zip -d openvino-models/musicgen
unzip musicgen-static-openvino/musicgen_small_stereo_openvino_models.zip -d openvino-models/musicgen
rm -rf musicgen-static-openvino
```

### Super Resolution
```bash
git clone --no-checkout https://huggingface.co/Intel/versatile_audio_super_resolution_openvino
cd versatile_audio_super_resolution_openvino
git checkout 9a97d7f128b22aea72e92862a3eccc310f88ac26
cd ..

unzip versatile_audio_super_resolution_openvino/versatile_audio_sr_base_openvino_models.zip -d openvino-models/audiosr
unzip versatile_audio_super_resolution_openvino/versatile_audio_sr_ddpm_basic_openvino_models.zip -d openvino-models/audiosr
unzip versatile_audio_super_resolution_openvino/versatile_audio_sr_ddpm_speech_openvino_models.zip -d openvino-models/audiosr
rm -rf versatile_audio_super_resolution_openvino
```

### Install models system-wide
```bash
sudo cp -R openvino-models /usr/local/lib/
```

---

## 10. Adding the desktop icon

```bash
cat > ~/.local/share/applications/audacity-openvino.desktop << 'EOF'
[Desktop Entry]
Name=Audacity (OpenVINO)
Comment=Audio Editor with OpenVINO AI Effects
Exec=/home/$USER/audacity-openvino/audacity-build/Release/bin/audacity
Icon=audacity
Terminal=false
Type=Application
Categories=Audio;AudioVideo;
EOF

cp ~/.local/share/applications/audacity-openvino.desktop ~/Desktop/
chmod +x ~/Desktop/audacity-openvino.desktop
```

---

## Troubleshooting

**System lags/freezes after the build**
Building OpenVINO and Audacity from source creates hundreds of thousands of files. KDE's file indexer (Baloo) will try to index all of them after the build, which on a HDD can cause severe I/O bottlenecks and system-wide freezes for multiple hours.

Fix: Exclude the build directories from Baloo and/or disable it entirely:
```bash

balooctl6 config excludeFolder ~/audacity-openvino
balooctl6 config excludeFolder ~/.cache/yay

# Or disable Baloo completely if you don't use KDE search
balooctl6 disable
```

**`Mismatch between the program and library build versions` fatal error**
This happens when you copy `mod-openvino.so` into the system-installed Audacity. The self-compiled Audacity uses wxWidgets 3.2, but the Arch `audacity` package uses wxWidgets 3.1.3 — they are binary incompatible. Always use the self-compiled Audacity binary at `~/audacity-openvino/audacity-build/Release/bin/audacity`. Do NOT copy the module into `/usr/lib/audacity/modules/`.

**`unsupported OS: arch` error in install scripts**
The official dependency installer scripts are Ubuntu-specific. Skip them entirely and use the `pacman` commands from Step 1 instead.

**`'uint16_t' has not been declared` / `bfloat16.hpp` errors during `make`**
OpenVINO 2024.6.0 was written for GCC 14. Arch ships GCC 15, which is stricter about implicit includes. The fix is to patch `bfloat16.hpp` to explicitly include `<cstdint>`. This is already handled in Step 2 above via the `sed` command before `make`.

**`onednn_gpu_build-configure` or `cpplint` errors during `make`**
Two separate issues: the GPU submodule (`onednn_gpu`) has the same CMake 4.x compatibility problem but runs as a separate process and doesn't inherit the policy fix. The `cpplint` errors are from a code style checker that is irrelevant for our purposes. Fix: disable GPU, NPU and cpplint via `-DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_NPU=OFF -DENABLE_CPPLINT=OFF`. GPU/NPU are not needed for audio processing — CPU inference is fully sufficient. These flags are already included in the cmake command above.

**`ONNX::onnx contains absl::absl_check but target was not found` error**
Arch's `onnx` package has broken CMake target exports — even with `abseil-cpp` installed, the targets are not correctly resolved. The fix is to temporarily remove the system `onnx` package (`sudo pacman -R onnx`) so OpenVINO falls back to its own bundled onnx. You can reinstall it after the build with `sudo pacman -S onnx`. This is already handled in Step 2 above.

**`Compatibility with CMake < 3.5 has been removed` error**
Arch Linux ships a very recent CMake (4.x) which dropped backwards compatibility with old submodule CMakeLists.txt files (e.g. `level-zero`). Add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to the cmake command. This is already included in the cmake command above.

**`FindOpenVINO.cmake` not found**
Make sure you have sourced the OpenVINO environment variables:
```bash
source ~/audacity-openvino/openvino_toolkit/installed/setupvars.sh
```
And that `-DOpenVINO_DIR` is passed correctly to cmake.

**`whisper` linked by target `mod-openvino` set to NOTFOUND**
Make sure `WHISPERCPP_ROOTDIR` is exported and points to the correct `installed/` directory from Step 5.

**Module loads but effects don't appear**
Make sure the AI models are installed at `/usr/local/lib/openvino-models/` (see Step 9).

**Missing ffmpeg module when starting Audacity**
If you are missing the ffmpeg module, simply install it

```bash
sudo pacman -S ffmpeg
```
---

*This guide was created as a community contribution based on the official Ubuntu 22.04 build instructions. Please report issues in the [openvino-plugins-ai-audacity issue tracker](https://github.com/intel/openvino-plugins-ai-audacity/issues).*
