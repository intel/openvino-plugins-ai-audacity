#!/bin/bash

target_dir="/usr/local/lib/openvino-models"

if [ ! -d "$target_dir" ]; then
    sudo mkdir -p "$target_dir"
fi

mkdir openvino-models
cd openvino-models

# download_and_move_files() can download and move individual files, if they are listed
# it downloads usually xml. and .bin files ; Not for archives (no unzip)
download_and_move_files() {
    local base_url="$1"
    shift
    local filenames=()
    while [ "$1" != "::" ]; do
        filenames+=("$1")
        shift
    done
    shift  # skip "--"
    local extensions=()
    while [ "$1" != "::" ]; do
        extensions+=("$1")
        shift
    done
    shift  # skip "--"
    local log_file="$1"
  
    for filename in "${filenames[@]}"; do
        for ext in "${extensions[@]}"; do
            local full_filename="${filename}.${ext}"
            
            if [ -f "$full_filename" ]; then
                echo "$full_filename already exists, skipping download. Moving the file to /usr/local/lib/openvino-modules."
                sudo mv "$full_filename" /usr/local/lib/openvino-models        
                echo "moved $full_filename to /usr/local/lib/openvino-models" >> "$log_file"
            else
                if [[ ! $answer =~ ^[YyAaNn]$ ]]; then
                  read -p "Do you want to download noise-suppression-denseunet files? (y/n) " answer
                fi
                if [[ $answer =~ ^[YyAa]$ ]]; then
                    echo "I try to download $full_filename."
                    if wget "${base_url}/${full_filename}"; then
                        echo "downloaded $full_filename" >> "$log_file"
                        sudo mv "$full_filename" /usr/local/lib/openvino-models        
                        echo "moved $full_filename to /usr/local/lib/openvino-models" >> "$log_file"
                    fi
                fi
            fi
        done
    done
}


# OPENVINO, OPENVINO TOOLS AND OPENVINO PLUGINS COMPILED

# WHISPER COMPILED

# AUDACITY COMPILED WITH OPENVINO MODULES

# MODULES FINISHED

# Since many of these models will come from huggingface repos, 
# let's make sure git lfs is installed

if ! command -v git-lfs &> /dev/null
then
  sudo apt install git-lfs
fi


# To actually use these modules, we need to generate / populate 
#  /usr/local/lib/ with the OpenVINO models that the plugins will look for. 

#************
#* MusicGen *
#************

# clone the HF repo
read -p "Do you want to clone the MusicGen HF repo? (y/n) " answer
if [[ $answer =~ ^[Yy] ]]; then
    git clone https://huggingface.co/Intel/musicgen-static-openvino
else
    echo "Skipped musicgen-static-openvino"
fi

folder_static="musicgen-static-openvino"

# Seznam souborů k ověření (s cestou k složce)
files_static=(
    "$folder_static/musicgen_small_enc_dec_tok_openvino_models.zip"
    "$folder_static/musicgen_small_mono_openvino_models.zip"
    "$folder_static/musicgen_small_stereo_openvino_models.zip"
)

for file in "${files_static[@]}"
do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Downloaded '$filename'." >> "$folder_static.log"
    fi
done

# @TODO: NOT NECESSARY!! clone the HF Facebook repo
read -p "Do you want to clone the MusicGen stereo HF Facebook repo cca 9GB? (y/n) This package is not needed for OpenVino!" answer
if [[ $answer =~ ^[YyAa]$ ]]; then
    git clone https://huggingface.co/facebook/musicgen-stereo-small
fi

# There is more files than that, but these are the biggest onenes!
folder_facebook="musicgen-stereo-small"
files_facebook=(
    "$folder_facebook/state_dict.bin"
    "$folder_facebook/model.safetensors"
    "$folder_facebook/pytorch_model.bin"
    "$folder_facebook/model.fp32.safetensors"
    "$folder_facebook/pytorch_model.fp32.bin"
    "$folder_facebook/tokenizer.json"
)

for file in "${files_facebook[@]}"
do
    if [ -f "$file" ]; then
        echo "File '$file' created." >> "$folder_facebook.log"
    fi
done

folder_static="musicgen-static-openvino"
zip_files=(
    "$folder_static/musicgen_small_enc_dec_tok_openvino_models.zip"
    "$folder_static/musicgen_small_mono_openvino_models.zip"
    "$folder_static/musicgen_small_stereo_openvino_models.zip"
)

log_files=(
    "static_encdec.log"
    "static_small.log"
    "static_stereo.log"
)

# Loop through each zip file, unzip it, log the extraction and delete the zip file
for i in "${!zip_files[@]}"; do
    zip_file="${zip_files[i]}"
    log_file="${log_files[i]}"
    if unzip "$zip_file" -d "musicgen"; then
        echo "extracted $(basename "$zip_file")" >> "$log_file"
        rm "$zip_file"
        echo "  --> deleted $(basename "$zip_file")" >> "$log_file"
    fi
done

#*************************
#* Whisper Transcription *
#*************************

# This section requires admin permissions

# clone the HF repo - https://huggingface.co/Intel/whisper.cpp-openvino-models/tree/main
read -p "Do you want to clone the Whisper HF repo using choice mode? (y/n) " answer
if [[ $answer =~ ^[YyAa]$ ]]; then
    # It's just too big!!!
    read -p "Download small and medium Whisper models using wget?" answer
    if [[ $answer =~ ^[YyAa]$ ]]; then
        if ! grep -q "downloaded ggml-base-models.zip" whisper_base.log; then
         if [ ! -f "ggml-base-models.zip" ]; then
            wget https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/main/ggml-base-models.zip
         fi
        fi
        if ! grep -q "downloaded ggml-medium-models.zip" whisper_medium.log; then
         if [ ! -f "ggml-medium-models.zip" ]; then
            wget https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/main/ggml-medium-models.zip
         fi
        fi
        if ! grep -q "ggml-small.en-tdrz-models.zip" whisper_small_en_tdrz.log; then
         if [ ! -f "ggml-small.en-tdrz-models.zip" ]; then
            wget https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/main/ggml-small.en-tdrz-models.zip
         fi
        fi
        if ! grep -q "ggml-small-models.zip" whisper_small.log; then
         if [ ! -f "ggml-small-models.zip" ]; then
            wget https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/main/ggml-small-models.zip
         fi
        fi
    else    
        read -p "Download small and medium Whisper models using git/git-lfs?" answer
        if [[ $answer =~ ^[YyAa]$ ]]; then
          git init
          git remote add origin https://huggingface.co/Intel/whisper.cpp-openvino-models
          git config core.sparseCheckout true
          if ! grep -q "downloaded ggml-base-models.zip" whisper_base.log; then
            if [ ! -f "ggml-base-models.zip" ]; then
                echo "ggml-base-models.zip" > .git/info/sparse-checkout
            fi
          fi
          if ! grep -q "downloaded ggml-medium-models.zip" whisper_medium.log; then
           if [ ! -f "ggml-medium-models.zip" ]; then
             echo "ggml-medium-models.zip" >> .git/info/sparse-checkout
           fi
          fi
          if ! grep -q "ggml-small.en-tdrz-models.zip" whisper_small_en_tdrz.log; then          
            if [ ! -f "ggml-small.en-tdrz-models.zip" ]; then
              echo "ggml-small.en-tdrz-models.zip" >> .git/info/sparse-checkout
            fi
          fi
          
          if ! grep -q "ggml-small-models.zip" whisper_small.log; then
           if [ ! -f "ggml-small-models.zip" ]; then
             echo "ggml-small-models.zip" >> .git/info/sparse-checkout
           fi
          fi
          git pull origin main
        else
            read -p "Download small, medium and large v3 Whisper models using git/git-lfs?" answer
            if [[ $answer =~ ^[YyAa]$ ]]; then
                  git init
                  git remote add origin https://huggingface.co/Intel/whisper.cpp-openvino-models
                  git config core.sparseCheckout true
                  if ! grep -q "downloaded ggml-base-models.zip" whisper_base.log; then
                     if [ ! -f "ggml-base-models.zip" ]; then
                          echo "ggml-base-models.zip" > .git/info/sparse-checkout
                     fi
                  fi
                  if ! grep -q "downloaded ggml-medium-models.zip" whisper_medium.log; then
                     if [ ! -f "ggml-medium-models.zip" ]; then
                          echo "ggml-medium-models.zip" >> .git/info/sparse-checkout
                     fi
                  fi
                  if ! grep -q "ggml-small-models.zip" whisper_small.log; then
                      if [ ! -f "ggml-small-models.zip" ]; then
                        echo "ggml-small-models.zip" >> .git/info/sparse-checkout
                      fi
                  fi
                  if [ ! -f "ggml-small.en-tdrz-models.zip" ]; then
                    echo "ggml-small.en-tdrz-models.zip" >> .git/info/sparse-checkout
                  fi
                  if ! grep -q "ggml-large-v3-models.zip" whisper_large_v3_models.log; then
                      if [ ! -f "ggml-large-v3-models.zip" ]; then
                        echo "ggml-large-v3-models.zip" >> .git/info/sparse-checkout
                      fi
                  fi
                  git pull origin main
            else
                read -p "Download all Whisper models including large models assumed (40-60GB) using git/git-lfs?" answer
                if [[ $answer =~ ^[YyAa]$ ]]; then
                  git clone https://huggingface.co/Intel/whisper.cpp-openvino-models
                fi
            fi
        fi
    fi
fi

# Unzip the files and log the extraction

folder_whisper="whisper.cpp-openvino-models"
target_dir="/usr/local/lib/openvino-models"
zip_files=(
    "$folder_whisper/ggml-base-models.zip"
    "$folder_whisper/ggml-small-models.zip"
    "$folder_whisper/ggml-small.en-tdrz-models.zip"
    "$folder_whisper/ggml-medium-models.zip"
    "$folder_whisper/ggml-large-v1-models.zip"
    "$folder_whisper/ggml-large-v2-models.zip"
    "$folder_whisper/ggml-large-v3-models.zip"
)

log_files=(
    "whisper_base.log"
    "whisper_small.log"
    "whisper_small_en_tdrz.log"
    "whisper_medium-models.log"
    "whisper_large-v1-models.log"
    "whisper_large-v2-models.log"
    "whisper_large-v3-models.log"
)

declare -A extracted_files
extracted_files["$folder_whisper/ggml-base-models.zip"]="ggml-base-encoder-openvino.bin ggml-base-encoder-openvino.xml ggml-base.bin"
extracted_files["$folder_whisper/ggml-small-models.zip"]="ggml-small-encoder-openvino.bin ggml-small-encoder-openvino.xml ggml-small.bin"

extracted_files["$folder_whisper/ggml-small.en-tdrz-models.zip"]="ggml-small.en-tdrz-encoder-openvino.bin ggml-small.en-tdrz-encoder-openvino.xml ggml-small.en-tdrz.bin"

extracted_files["$folder_whisper/ggml-medium-models.zip"]="ggml-medium-encoder-openvino.bin ggml-medium-encoder-openvino.xml ggml-medium.bin"

# @TODO: To check the file names for the large files - near version
extracted_files["$folder_whisper/ggml-large-v1-models.zip"]="ggml-large-v1-encoder-openvino.bin ggml-large-v1-encoder-openvino.xml ggml-large-v1.bin"

extracted_files["$folder_whisper/ggml-large-v2-models.zip"]="ggml-large-v2-encoder-openvino.bin ggml-large-v2-encoder-openvino.xml ggml-large-v2.bin"

extracted_files["$folder_whisper/ggml-large-v3-models.zip"]="ggml-large-v3-encoder-openvino.bin ggml-large-v3-encoder-openvino.xml ggml-large-v3.bin"

# Function to check if all files from a zip archive are present in the target directory
check_and_extract() {
    local archive_folder="$1"
    local target_directory="$2"
    local zip_file="$3"
    local log_file="$4"
    local all_files_extracted=true

    for file in ${extracted_files[$zip_file]}; do
        if ! grep -q "extracted $file" "$log_file"; then
            all_files_extracted=false
            break
        fi
    done

    if [ "$all_files_extracted" = false ]; then    
        echo "Extracting $zip_file..."
        for file in ${extracted_files[$zip_file]}; do
            echo "I look for $file ..."
            if [ ! -f "$target_directory/$file" ]; then
                if [ -f "$archive_folder/$file" ]; then
                    echo "Moving $archive_folder/$file to $target_directory/$file"
                    sudo mv "$archive_folder/$file" "$target_directory/$file" && echo "moved $file from $(basename "$zip_file")" >> "$log_file"
                else
                    echo "unziping with options -qq -n $zip_file , file $file to $target_directory ..."
                    sudo unzip -qq -n "$zip_file" "$file" -d "$target_directory" && echo "extracted and moved $file from $(basename "$zip_file")" >> "$log_file"
                fi
            else
                echo "already extracted $file from $(basename "$zip_file")" >> "$log_file"
            fi
        done
    else
        echo "All files should be extracted. Check the logs."
        read -p "Do you want to delete the archive $zip_file that has been extracted to $target_dir? (y/n)" answer
        if [[ $answer =~ ^[Yy] ]]; then
            echo "Deleting archive..."
            rm "$zip_file"
        fi
    fi
}

# Note Special Variable Syntax of Exclamation mark "!"
# ${!zip_files[@]}: Retrieves all indices of the zip_files array.
# Looping through indices: for i in "${!zip_files[@]}" sequentially iterates through indices of zip_files.
# The ! in ${!zip_files[@]} fetches array indices

echo "----------------------------------------------"
echo "<<<<<<<<< Detecting Whisper files >>>>>>>>>>>>"
echo "----------------------------------------------"
for i in "${!zip_files[@]}"; do
    zip_file="${zip_files[i]}"
    log_file="${log_files[i]}"
    if [ -f "$zip_file" ]; then
        echo "downloaded $(basename "$zip_file")" >> "$log_file"
        check_and_extract "$folder_whisper" "$target_dir" "$zip_file" "$log_file"
    fi
done


#********************
#* Music Separation *
#********************

touch demucs.log
if ! grep -q "demucs-openvino/htdemucs_v4.bin moved to /usr" demucs.log && \
   ! grep -q "demucs-openvino/htdemucs_v4.xml moved to /usr" demucs.log; then
    # clone the HF repo
    read -p "Do you want to clone the Demucs HF repo? (y/n) " answer
    if [[ $answer =~ ^[YyAa]$ ]]; then
        if [ ! -f "demucs-openvino/htdemucs_v4.bin" ] || [ ! -f "demucs-openvino/htdemucs_v4.xml" ]; then
        git clone https://huggingface.co/Intel/demucs-openvino
        fi
    fi
fi

# Copy the demucs OpenVINO IR files
# Copy the demucs OpenVINO IR files
echo "Moving demucs files ... htdemucs_v4.bin & htdemucs_v4.xml"
if sudo mv demucs-openvino/htdemucs_v4.bin /usr/local/lib/openvino-models && sudo mv demucs-openvino/htdemucs_v4.xml /usr/local/lib/openvino-models; then
  # Now that the required models are copied, remove non-log files
  # find demucs-openvino -type f ! -name '*.log' -exec rm -f {} +
  echo "demucs-openvino/htdemucs_v4.bin moved to /usr/local/lib/openvino-models" >> demucs.log
  echo "demucs-openvino/htdemucs_v4.xml moved to /usr/local/lib/openvino-models" >> demucs.log
fi


# Now that the required models are extracted, feel free to delete the cloned 'demucs-openvino' directory.

#*********************
#* Noise Suppression *
#*********************

# Clone the deepfilternet HF repo
read -p "Do you want to clone the DeepFilterNet HF repo? (y/n) " answer
if [[ $answer =~ ^[YyAa]$ ]]; then
    git clone https://huggingface.co/Intel/deepfilternet-openvino
    if [ -f deepfilternet-openvino/deepfilternet2.zip ]; then
      echo "downloaded deepfilternet2.zip" > deepfilter.log
    fi
    if [ -f deepfilternet-openvino/deepfilternet3.zip ]; then
      echo "downloaded deepfilternet3.zip" > deepfilter.log
    fi
fi

touch deepfilter.log
# Check and extract deepfilternet2.zip
if grep -q "extracted deepfilternet2.zip" deepfilter.log; then
   rm deepfilternet-openvino/deepfilternet2.zip
else
   if unzip deepfilternet-openvino/deepfilternet2.zip -d deepfilternet2; then
     echo "extracted deepfilternet2.zip" >> deepfilter.log
     rm deepfilternet-openvino/deepfilternet2.zip
     echo "deleted deepfilternet2.zip" >> deepfilter.log
     sudo mv deepfilternet2 /usr/local/lib/openvino-models
     echo "moved deepfilternet2 to /usr/local/lib/openvino-models" >> deepfilter.log
   fi
fi

# Check and extract deepfilternet3.zip
if grep -q "extracted deepfilternet3.zip" deepfilter.log; then
    if unzip deepfilternet-openvino/deepfilternet3.zip -d deepfilternet3; then
      echo "extracted deepfilternet3.zip" >> deepfilter.log
      rm deepfilternet-openvino/deepfilternet3.zip
      echo "deleted deepfilternet3.zip" >> deepfilter.log
      sudo mv deepfilternet3 /usr/local/lib/openvino-models
      echo "moved deepfilternet to /usr/local/lib/openvino-models" >> deepfilter.log
    fi
else
    rm deepfilternet-openvino/deepfilternet3.zip
fi

####################
# NOISE SUPPRESSION
####################
# DOWNLOAD INDIVIDUAL FILES
# For noise-suppression-denseunet-ll-0001, we can wget IR from openvino repo

base_url="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16"
filenames=("noise-suppression-denseunet-ll-0001") # you can add more file base names here
extensions=("xml" "bin") # it will try to download all specified files in .xml and .bin
log_file="noise_suppression.log"
download_and_move_files "$base_url" "${filenames[@]}" :: "${extensions[@]}" :: "$log_file"

