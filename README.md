# OpenVINO™ AI Plugins for Audacity* :metal:
![](doc/assets/openvino_ai_plugins.png)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)  

A set of AI-enabled effects, generators, and analyzers for [Audacity®](https://www.audacityteam.org/). These AI features run 100% locally on your PC :computer: -- no internet connection necessary! [OpenVINO™](https://github.com/openvinotoolkit/openvino) is used to run AI models on supported accelerators found on the user's system such as CPU, GPU, and NPU.

- [**Music Separation**](doc/feature_doc/music_separation/README.md):musical_note: -- Separate a mono or stereo track into individual stems -- Drums, Bass, Vocals, & Other Instruments. 
- [**Music Style Remix**](doc/feature_doc/music_style_remix/README.md):cd: -- Uses Stable Diffusion to alter a mono or stereo track using a text prompt.
- [**Noise Suppression**](doc/feature_doc/noise_suppression/README.md):broom: -- Removes background noise from an audio sample.
- [**Music Generation**](doc/feature_doc/music_generation/README.md):notes: -- Uses Stable Diffusion to generate snippets of music from a text prompt.
- [**Whisper Transcription**](doc/feature_doc/whisper_transcription/README.md):microphone: -- Uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) to generate a label track containing the transcription or translation for a given selection of spoken audio or vocals.

## Installation :floppy_disk: 
  Go [here](https://github.com/intel/openvino-plugins-ai-audacity/releases) to find installation packages & instructions for the latest Windows release. 

## Build Instructions :hammer:
  - [Windows Build Instructions](doc/build_doc/windows/README.md)  
  - [Linux Build Instructions](doc/build_doc/linux/README.md)

## Help, Feedback, & Bug Reports 🙋‍♂️
  We welcome you to submit an issue [here](https://github.com/intel/openvino-plugins-ai-audacity/issues) for
  * Questions
  * Bug Reports
  * Feature Requests
  * Feedback of any kind -- how can we improve this project?
    
## Contribution :handshake:
  Your contributions are welcome and valued, no matter how big or small. Feel free to submit a pull-request!

## Acknowledgements :pray:
* Audacity® development team & Muse Group-- Thank you for your support!  
* Audacity® GitHub -- https://github.com/audacity/audacity  
* Whisper transcription & translation analyzer uses whisper.cpp (with OpenVINO™ backend): https://github.com/ggerganov/whisper.cpp  
* Music Generation & Music Style Remix use Riffusion's UNet model, Riffusion pipelines that were ported to C++ from this project: https://github.com/riffusion/riffusion    
* Music Separation effect uses Meta's Demucs v4 model (https://github.com/facebookresearch/demucs), which has been ported to work with OpenVINO™  
* Noise Suppression uses noise-suppression-denseunet-ll model from OpenVINO™'s Open Model Zoo: https://github.com/openvinotoolkit/open_model_zoo  

## Disclaimer :warning:
Stable Diffusion & Riffusion's data model is governed by the Creative ML Open Rail M license, which is not an open source license.
https://github.com/CompVis/stable-diffusion. Users are responsible for their own assessment whether their proposed use of the project code and model would be governed by and permissible under this license.
