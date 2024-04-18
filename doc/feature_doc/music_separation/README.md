# OpenVINO™ Music Separation :musical_note:

This feature allows a user to separate a mono or stereo track into individual stems -- Drums, Bass, Vocals, & Other instruments.  

You can find this feature in the **Effect** menu:  
![music_sep_menu](https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/3fada2b8-d5d1-45af-bae5-94b6d106a5ff)

To start, select a portion of a mono or stereo track, and select *OpenVINO Music Separation* from the menu.  

## Description of properties
See below for a description of the properties that can be set for this effect:  
![music_sep_dialog40](https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/8b6b2b5f-36b8-4250-9454-8ab543fc6199)
1. **Separation Mode**: Here you can either choose *2-Stem*, which will produce 2 new tracks (Instrumental, Vocals), or you can select *4-Stem* which will produce 4 new tracks (Drums, Bass, Other Instruments, Vocals). 
2. **OpenVINO Inference Device**: The OpenVINO™ device that will be used to run the music separation AI model (demucs v4)
3. **Device Details**: Clicking this button will give more detailed information about your devices, and device-mapping. For example, this can be useful if you have multiple GPUs and to easily understand which is mapped to 'GPU.0', and 'GPU.1', etc.
4. **Advanced** Checkbox: Click this to view advanced settings for this effect
5. **Shifts**: This effect works by running the htdemucs pipeline *Shifts* times (shifting the input audio randomly for each one), and then combining the results. A higher value *may* produce better results, but note that the processing timewill scale linearly with respect to the number of shifts.

After clicking *Apply*, you'll see this dialog window pop up:  
![](loading.png)

At this stage, the Music Separation AI model is getting loaded to the chosen device (e.g. CPU, GPU, etc.). This usually takes 10 to 30 seconds if it's the first time running with this device after installing these plugins since it needs to compile the model specifically for the device you've chosen. This compiled model will be cached on disk though -- so it should run much faster the next time that it is loaded.

When completed, either 2 or 4 new tracks will show up in the Audacity workspace. For example in *4-Stem* mode, the generated tracks would look something like the following. 
![](output.png)

**Note**: The generated tracks will have "-Drums", "-Bass", "-Vocals", etc. appended to the track name so that you can easily tell them apart.
