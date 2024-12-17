# OpenVINO™ Super Resolution :sparkles:

This feature performs high-quality audio super-resolution, enhancing diverse audio types to 24kHz bandwidth and 48kHz sampling rate for improved fidelity and versatility. It is an port of the python project found here: https://github.com/haoheliu/versatile_audio_super_resolution

It can be found under the **Effect** menu:  
<img width="315" alt="audio_sr_menu" src="https://github.com/user-attachments/assets/56056469-3914-416c-aa85-3035079c7050">

To start, select a portion of a mono or stereo track, and select *OpenVINO Super Resolution* from the menu.

## Description of properties
See below for a description of the properties that can be set for this effect:  
![audio_sr_settings](https://github.com/user-attachments/assets/4a5fe1f2-0eed-4974-94e7-bee02e8352b4)
1. **Unload Models**: For the first 'Apply' after Audacity is opened, the AI models are loaded into memory. Once loaded, they are kept in memory to reduce the time it takes to run Super Resolution again (which is nice for experimentation). Once you are done and happy with the results, you can click 'Unload Models' to free up system memory.
2. **Model**: Select between *Basic (General)* and *Speech* models.
   - Basic (General): Use for enhancing all types of audio including music and environmental sounds.
   - Speech: Optimized for enhancing audio with isolated speech.
3. **Device Selection**: The set of OpenVINO™ devices that will be used to run the various stages of the Super Resolution Diffusion pipeline.
4. **Device Details**: Clicking this button will give more detailed information about your devices, and device-mapping. For example, this can be useful if you have multiple GPUs and to easily understand which is mapped to 'GPU.0', and 'GPU.1', etc.
5. **Normalize Output RMS**: Check this if you want the resultant enhanced audio to be normalized to match the RMS of the input audio.
6. **Advanced Options**: Click this to view advanced settings for this effect
7. **Chunk Size**: Choose between 10.24 & 5.12 seconds. The audio is processed in chunks of this size (with some overlap + crossfade). A smaller chunk size (5.12 seconds) *may* produce better results in some circumstances, but will increase processing time.
8. **Seed**: Used to initialize the RNG (random noise generator) for the diffusion pipeline. If left blank, an arbitrary value is chosen as a seed, and will be different each time you run this effect. This can be a good thing for experimentation. To generate consistent results, or to recreate something previously generated, the seed should be explicitly set.
9. **Steps**: The number of DDPM inference steps used to enahnce each *chunk* of audio. In general, the higher this is set, the higher quality the enhanced audio may be... at least up to a certain point.
10. **Guidance Scale**:  Used during the diffusion process to control the trade-off between fidelity (how true the output is to the input) and creativity (how much the model enhances or modifies the audio).
    - A higher guidance scale emphasizes the model's adherence to the input features, leading to more faithful and accurate reconstruction of the input audio during the super-resolution process.
    - A lower guidance scale allows the model to be more creative, potentially introducing novel enhancements or artifacts that might not closely match the input.
