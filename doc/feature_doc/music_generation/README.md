# OpenVINO™ Music Generation & Continuation :notes:

This feature allows a user to generate snippets of music from a text prompt, as well as generate a continuation from an already exiting snippet of music.  

It can be found under the **Generate** menu:  
<img width="518" alt="ov_music_gen_menu" src="https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/7be98a8c-f345-4700-9b20-9733c097870b">

## Description of Properties:
![musicgen_dialog_40](https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/876b95f3-c146-45ef-8f9d-5ca25a10931c)

1. **Unload Models**: For the first 'Generate' after Audacity is opened, the AI models are loaded into memory. Once loaded, they are kept in memory to reduce the time it takes for successive calls to 'Generate' (which is nice for experimentation). Once you are done generating, and happy with the results, you can click 'Unload Models' to free up system memory.
2. **Duration**: The desired duration of the generated audio snippet.
3. **Model Selection**: This is where you can select between the various MusicGen models that are detected to have been installed. Note that these models are selected / downloaded at installation time.
   - fp16 (16-bit floating-point) models (in general) produce better quality than int8 (8-bit integer) compressed models. But, int8 models consume less system memory and run faster than fp16 models on most devices.
   - mono models natively produce a single channel of audio (32khz), stereo models natively produce 2 stereo channels (also 32khz).
4. **Prompt**: This is used to describe the type of music to generate. It works best to describe 1.
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

## Music Generation:
https://github.com/intel/openvino-plugins-ai-audacity/assets/107415876/7b975dc8-16a7-4e31-952c-9e367be51ee7

