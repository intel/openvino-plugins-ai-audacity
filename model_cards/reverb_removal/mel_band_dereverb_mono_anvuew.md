# MelBandRoformer Dereverb Mono (@anvuew)

A MelBandRoformer-based model trained to extract / remove reverb from vocals. 

This particular (mono) variant seems to work well for spoken audio as well, even though it's primarily trained on vocal tracks (i.e. music). In the case of spoken audio tracks, you can try running directly on a mono or stereo track. 

We've noticed that for some spoken tracks with only *minor* reverb present, sometimes this model doesn't end up removing the reverb. This might seem odd, but in these cases you can try to *add* some additional reverb to the source track (**Effects->Delay and Reverb->Reverb..**), and then re-apply the **Reverb Removal** effect.

As this is a MelBandRoformer model, so expect long processing times.

This model was trained by [@anvuew](https://huggingface.co/anvuew).

The original pytorch checkpoint / config were downloaded from HuggingFace here: [https://huggingface.co/anvuew/dereverb_mel_band_roformer](https://huggingface.co/anvuew/dereverb_mel_band_roformer)

For use with [OpenVINO AI Plugins for Audacity](https://github.com/intel/openvino-plugins-ai-audacity), the pytorch source models were converted to OpenVINO IR format, and stored here: TODO

License: [GPL-3.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/gpl-3.0.md)