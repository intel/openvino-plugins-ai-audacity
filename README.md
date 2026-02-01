# OpenVINO™ AI Plugins for Audacity* :metal:
![updated_audacity_top_level](https://github.com/user-attachments/assets/f7c911db-fb98-43ff-972d-f66473b4c921)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Mentioned in Awesome OpenVINO](https://awesome.re/mentioned-badge-flat.svg)](https://github.com/openvinotoolkit/awesome-openvino)

A set of AI-enabled effects, generators, and analyzers for [Audacity®](https://www.audacityteam.org/). These AI features run 100% locally on your PC :computer: -- no internet connection necessary! [OpenVINO™](https://github.com/openvinotoolkit/openvino) is used to run AI models on supported accelerators found on the user's system such as CPU, GPU, and NPU.

- [**Music Separation**](doc/feature_doc/music_separation/README.md):musical_note: -- Separate a mono or stereo track into individual stems -- Drums, Bass, Vocals, & Other Instruments. 
- [**Noise Suppression**](doc/feature_doc/noise_suppression/README.md):broom: -- Removes background noise from an audio sample.
- [**Whisper Transcription**](doc/feature_doc/whisper_transcription/README.md):microphone: -- Uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) to generate a label track containing the transcription or translation for a given selection of spoken audio or vocals.
- [**Super Resolution**](doc/feature_doc/super_resolution/README.md):sparkles: -- Upscales and enriches audio for improved clarity and detail.

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

## Acknowledgements & Citations :pray:
* Audacity® development team & Muse Group-- Thank you for your support!  
* Audacity® GitHub -- https://github.com/audacity/audacity  
* Whisper transcription & translation analyzer uses whisper.cpp (with OpenVINO™ backend): https://github.com/ggerganov/whisper.cpp  
* Music Separation effect uses Meta's Demucs v4 model (https://github.com/facebookresearch/demucs), which has been ported to work with OpenVINO™
* Noise Suppression:
  * noise-suppression-denseunet-ll:  from OpenVINO™'s Open Model Zoo: https://github.com/openvinotoolkit/open_model_zoo   
  * DeepFilterNet2 & DeepFilterNet3:
    * Ported the models & pipeline from here: https://github.com/Rikorose/DeepFilterNet  
    * We also made use of @grazder's fork / branch (https://github.com/grazder/DeepFilterNet/tree/torchDF-changes) to better understand the Rust implementation, and so we also based some of our C++ implementation on ```torch_df_offline.py``` found here.  
    * Citations:
      ```bibtex
      @inproceedings{schroeter2022deepfilternet2,
      title = {{DeepFilterNet2}: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio},
      author = {Schröter, Hendrik and Escalante-B., Alberto N. and Rosenkranz, Tobias and Maier, Andreas},
      booktitle={17th International Workshop on Acoustic Signal Enhancement (IWAENC 2022)},
      year = {2022},
      }
        
      @inproceedings{schroeter2023deepfilternet3,
      title = {{DeepFilterNet}: Perceptually Motivated Real-Time Speech Enhancement},
      author = {Schröter, Hendrik and Rosenkranz, Tobias and Escalante-B., Alberto N. and Maier, Andreas},
      booktitle={INTERSPEECH},
      year = {2023},
      }
      ```
* Super Resolution feature is a port (python -> C++, pytorch -> OpenVINO IR) of this project: https://github.com/haoheliu/versatile_audio_super_resolution
  * Citations:
    ```bibtex
    @article{liu2023audiosr,
    title={{AudioSR}: Versatile Audio Super-resolution at Scale},
    author={Liu, Haohe and Chen, Ke and Tian, Qiao and Wang, Wenwu and Plumbley, Mark D},
    journal={arXiv preprint arXiv:2309.07314},
    year={2023}
    }
    ```
* [OpenVINO™ Notebooks](https://github.com/openvinotoolkit/openvino_notebooks) -- We have learned a lot from this awesome set of python notebooks, and are still using it to learn latest / best practices for implementing AI pipelines using OpenVINO™!
  
