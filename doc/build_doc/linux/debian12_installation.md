# **Audacity OpenVINO Module Installation Guide (Debian 12)**

This guide details the process of **building and installing Audacity with OpenVINO support** on **Debian 12**.

---

## **📌 Overview**
This guide walks through:
✔️ Installing **dependencies**  
✔️ Setting up **OpenVINO, LibTorch, and Whisper.cpp**  
✔️ **Building Audacity (vanilla version)**  
✔️ **Adding the OpenVINO module** and **rebuilding Audacity**  
✔️ **Installing OpenVINO models** for AI-powered features  

**📁 Default Working Directory:** `~/audacity-openvino/`  
_All installation files and builds will be placed here._

---

## **🛠 Step 1: Install Required Dependencies**
First, update your system and install necessary packages:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git python3-pip python3-venv \
  libgtk2.0-dev libasound2-dev libjack-jackd2-dev uuid-dev \
  ocl-icd-opencl-dev opencl-c-headers opencl-clhpp-headers \
  libglib2.0-dev libpango1.0-dev libfontconfig-dev \
  libfreetype-dev libharfbuzz-dev libjpeg-dev libpng-dev libtiff-dev \
  libxrender-dev libxext-dev libxi-dev libxrandr-dev unzip
```

---

## **🐍 Step 2: Set Up Python Virtual Environment & Install Conan**
Debian 12 enforces **PEP 668**, which prevents direct `pip` installations in the system Python environment. To work around this, use a **virtual environment**.

```bash
mkdir -p ~/audacity-openvino
cd ~/audacity-openvino
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install conan
```
✅ **Verify Conan installation**:
```bash
conan --version
```

---

## **🔄 Step 3: Install OpenVINO Toolkit**
```bash
cd ~/audacity-openvino
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/linux/l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64.tgz
tar xvf l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64.tgz
cd l_openvino_toolkit_*/install_dependencies/
sudo -E ./install_openvino_dependencies.sh
cd ..
source setupvars.sh
```

🔹 **Make OpenVINO available in every terminal session**:
```bash
echo 'source ~/audacity-openvino/l_openvino_toolkit_*/setupvars.sh' >> ~/.bashrc
source ~/.bashrc
```

---

## **🔥 Step 4: Install LibTorch**
LibTorch is required for many AI pipelines.

```bash
cd ~/audacity-openvino
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.1+cpu.zip
```

🔹 **Set environment variables**:
```bash
echo 'export LIBTORCH_ROOTDIR=~/audacity-openvino/libtorch' >> ~/.bashrc
echo 'export CMAKE_PREFIX_PATH=${LIBTORCH_ROOTDIR}' >> ~/.bashrc
source ~/.bashrc
```

---

## **🎤 Step 5: Build Whisper.cpp (for Speech-to-Text)**
```bash
cd ~/audacity-openvino
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
git checkout v1.5.4
cd ..

mkdir whisper-build
cd whisper-build
cmake ../whisper.cpp/ -DWHISPER_OPENVINO=ON
make -j$(nproc)
cmake --install . --config Release --prefix ./installed
```

🔹 **Set environment variables**:
```bash
echo 'export WHISPERCPP_ROOTDIR=~/audacity-openvino/whisper-build/installed' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${WHISPERCPP_ROOTDIR}/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## **🎵 Step 6: Build Audacity (Vanilla Version)**
```bash
cd ~/audacity-openvino
git clone https://github.com/audacity/audacity.git
cd audacity
git checkout release-3.7.1
cd ..

mkdir audacity-build
cd audacity-build
cmake -G "Unix Makefiles" ../audacity -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

✅ **Run Audacity to Test**:
```bash
./Release/bin/audacity
```

---

## **🧩 Step 7: Add OpenVINO Module to Audacity**
```bash
cd ~/audacity-openvino
git clone https://github.com/intel/openvino-plugins-ai-audacity.git
cp -r openvino-plugins-ai-audacity/mod-openvino ~/audacity-openvino/audacity/modules/
```

### **Modify `CMakeLists.txt`**
Edit **`~/audacity-openvino/audacity/modules/CMakeLists.txt`**:

```bash
nano ~/audacity-openvino/audacity/modules/CMakeLists.txt
```

Add this **after the `foreach` loop**:
```cmake
add_subdirectory(mod-openvino)
```

💾 **Save & Exit** (`CTRL+X`, `Y`, `ENTER`).

---

## **🔄 Step 8: Rebuild Audacity with OpenVINO**
```bash
cd ~/audacity-openvino/audacity-build
cmake -G "Unix Makefiles" ../audacity -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

✅ **Check that OpenVINO module is built**:
```bash
ls Release/lib/audacity/modules/
```
If `mod-openvino.so` is listed, you're good to go!

✅ **Run Audacity**:
```bash
./Release/bin/audacity
```

Then:
- **Edit → Preferences → Modules**
- Locate **mod-openvino**
- Change **New** → **Enabled**
- **Restart Audacity**

---

## **📥 Step 9: Install OpenVINO Models**
```bash
mkdir ~/audacity-openvino/openvino-models
cd ~/audacity-openvino/
sudo apt install git-lfs
```

#### **Download Required Models**
```bash
#************
#* MusicGen *
#************
mkdir openvino-models/musicgen

# clone the HF repo
git clone --no-checkout https://huggingface.co/Intel/musicgen-static-openvino
cd musicgen-static-openvino
git checkout b2ad8083f3924ed704814b68c5df9cbbf2ad2aae
cd ..

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
git clone --no-checkout https://huggingface.co/Intel/demucs-openvino
cd demucs-openvino
git checkout 97fc578fb57650045d40b00bc84c7d156be77547
cd ..

# Copy the demucs OpenVINO IR files
cp demucs-openvino/htdemucs_v4.bin openvino-models
cp demucs-openvino/htdemucs_v4.xml openvino-models

# Now that the required models are extracted, feel free to delete the cloned 'demucs-openvino' directory.
rm -rf demucs-openvino

#*********************
#* Noise Suppression *
#*********************

# Clone the deepfilternet HF repo
git clone --no-checkout https://huggingface.co/Intel/deepfilternet-openvino
cd deepfilternet-openvino
git checkout 995706bda3da69da0825074ba7dbc8a78067e980
cd ..

# extract deepfilter2 models
unzip deepfilternet-openvino/deepfilternet2.zip -d openvino-models

# extract deepfilter3 models
unzip deepfilternet-openvino/deepfilternet3.zip -d openvino-models

# For noise-suppression-denseunet-ll-0001, we can wget IR from openvino repo
cd openvino-models
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16/noise-suppression-denseunet-ll-0001.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16/noise-suppression-denseunet-ll-0001.bin
cd ..

#*********************
#* Super Resolution *
#*********************

# clone the HF repo
git clone --no-checkout https://huggingface.co/Intel/versatile_audio_super_resolution_openvino
cd versatile_audio_super_resolution_openvino
git checkout 9a97d7f128b22aea72e92862a3eccc310f88ac26
cd ..

# unzip the 'base' set of models into audiosr
unzip versatile_audio_super_resolution_openvino/versatile_audio_sr_base_openvino_models.zip -d openvino-models/audiosr

# unzip the basic ddpm model
unzip versatile_audio_super_resolution_openvino/versatile_audio_sr_ddpm_basic_openvino_models.zip -d openvino-models/audiosr

# unzip the speech ddpm model
unzip versatile_audio_super_resolution_openvino/versatile_audio_sr_ddpm_speech_openvino_models.zip -d openvino-models/audiosr

# Now that the required models are extracted, feel free to delete the cloned 'versatile_audio_super_resolution_openvino' directory.
rm -rf versatile_audio_super_resolution_openvino 
```

#### **Move Models to System Directory**
```bash
sudo cp -R ~/audacity-openvino/openvino-models /usr/local/lib/
```

---

## **🚀 Final Verification**
```bash
cd ~/audacity-openvino/audacity-build
./Release/bin/audacity
```
Go to **Modules**, ensure OpenVINO is **Enabled**, and restart.

🎉 **Installation is Complete!** 🚀  
