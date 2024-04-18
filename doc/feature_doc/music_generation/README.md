# OpenVINO™ Music Generation & Continuation :notes:

This feature allows a user to generate snippets of music from a text prompt, as well as generate a continuation from an already existing snippet of music.  

It can be found under the **Generate** menu:  
<img width="518" alt="ov_music_gen_menu" src="https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/7be98a8c-f345-4700-9b20-9733c097870b">

## Description of Properties:
![musicgen_dialog_40](https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/876b95f3-c146-45ef-8f9d-5ca25a10931c)

1. **Unload Models**: For the first 'Generate' after Audacity is opened, the AI models are loaded into memory. Once loaded, they are kept in memory to reduce the time it takes for successive calls to 'Generate' (which is nice for experimentation). Once you are done generating, and happy with the results, you can click 'Unload Models' to free up system memory.
2. **Duration**: The desired duration of the generated audio snippet.
3. **Model Selection**: This is where you can select between the various MusicGen models that are detected to have been installed. Note that these models are selected / downloaded at installation time.
   - fp16 (16-bit floating-point) models (in general) produce better quality than int8 (8-bit integer) compressed models. But, int8 models consume less system memory and run faster than fp16 models on most devices.
   - mono models natively produce a single channel of audio (32khz), stereo models natively produce 2 stereo channels (also 32khz).
4. **Prompt**: This is used to describe the type of music to generate. 
5. **Device Selection**: The set of OpenVINO™ devices that will be used to run the various stages of the MusicGen LLM pipeline. The pipeline is dominated by many decode (token generation) operations, so changing the Decode devices here will have the most impact on performance.
6. **Device Details**: Clicking this button will give more detailed information about your devices, and device-mapping. For example, this can be useful if you have multiple GPUs and to easily understand which is mapped to 'GPU.0', and 'GPU.1', etc.
7. **Seed**: Used to initialize the RNG (random noise generator). If left blank, an arbitrary value is chosen as a seed, and will be different each time you click 'Generate', and so generated results may greatly vary for each 'Generate'. And this can be a good thing for experimentation. To generate consistent results, or to recreate something previously generated, the seed should be set.
8. **Guidance Scale**: A value that represents how much the generation will adhere to your text prompt. A higher value can encourage the model to adhere closer to the prompt, but can decrease audio quality. Recommended to set between 2 and 4.
9. **TopK**: This is used to fine-tune the 'randomness' in the generated output, and can be used to shift the balance between coherence & creativity in the generated output. For more coherent, structured musical pieces, you might opt for a lower 'topK' value, such as 50. For more experimental, diverse compositions, a higher 'topK' such as 250 (or higher!) can be used. 
10. **Context Length**: This is used in a couple of ways:
    - If the duration of the generated audio snippet is longer than 20 seconds, the pipeline will use the last *N* seconds of audio as context to initialize the model to produce the next audio snippet.
    - If Audio Continuation is enabled, this controls the (maximum) amount of audio that will be used as context to generate a continuation of.
12. **Audio Continuation** Options: If applicable, these check boxes enable use of *Audio Continuation* features. See [here] for some sample videos that demonstrate use of these features.
    - Audio Continuation: If checked, audio continuation is enabled, meaning that some already exisiting snippet of music will be used to 'kick start' generation.
    - Audio Continuation on New Track: If checked, the generated audio along with the re-EnCodec'ed source audio will be added on a new track.

    **Note**: You'll find that this plugin makes some educated guesses and may grey-out or check these by default.
    - *They will be Greyed out when...* you haven't selected any existing audio, or if the selected track is empty. It's also greyed out if you've highlighted existing audio on more than 1 track.
    - *Audio Continuation will be checked by default when...* you have selected some existing chunk of audio, or if the the selection marker is sitting at the *end* of some existing chunk of audio.
    - *Audio Continuation on New Track will be checked by default when...* you've highlighted some chunk of audio that isn't at the end of the track. 
    

## Video Demonstrations

### Music Generation:
The following video demonstrates a simple text-to-music flow.

https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/7b975dc8-16a7-4e31-952c-9e367be51ee7

### Audio Continuation (On same track):
The following video demonstrates audio continuation, such that audio is generated into an existing track. This is very useful in cases where you want to extend previously generated audio segments.  

To do this, click the end-point of the track, and go to 'OpenVINO Music Generation' from the Generators menu. You'll see that the 'Audio Continuation' has been selected for you.

https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/8db00517-5fff-4309-ac95-b06907960fdb

### Audio Continuation (On new track): 
The following video demontrates audio continuation, such that audio (and re-Encodec'ed context) is generated into a new track. In this case, the existing audio track / selection is *not* something that was generated.  

To do this, select a portion of audio from a single track, and go to  'OpenVINO Music Generation' from the Generators menu. You'll see that the 'Audio Continuation' has been selected for you, as well as 'Audio Continuation on new track'.

https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/8bcecb7d-5a76-45fc-9b56-ed1cd731aef8


## Tips and Tricks

:bulb: **A generated segment is labeled with it's properties**: You'll notice that the generated audio snippet is labeled with the properties used to generate it:  
    <img width="402" alt="labeled_track" src="https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/33027a4d-3546-4e61-8450-18fc8eae96f4">

  This contains all of the information you need to recreate it, like seed, models used, devices, etc.
    
:bulb: **Generate short snippets when experimenting**: Setting the duration to something short, like 5 seconds, can save a lot of time when experimenting with prompts, models, devices, and other settings. If you generate a 5 second segment that you like, you can always generate something longer with the same seed, or use audio continuation to extend it!

:bulb: **Generated Music fades into silence, or transitions into noise is typical**: It's fairly normal for the audio generation to do something weird like fade into silence, or transition into something really weird sounding, or noise. If you liked what was generated before it did something weird, you can always delete the portion of audio that went astray (faded to silence, transitioned to noise, etc.) and then use audio continuation to pick up where the 'good' segment left off, and hopefully have more luck the second time.
