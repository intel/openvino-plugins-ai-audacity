# MelBandRoformer Vocals (@KimberleyJensen)

A MelBandRoformer-based vocal extraction model that excels at extracting vocals from input tracks. It's a heavier model as compared with *Demucs-v4*, so expect longer processing times. It might be worth the wait though, as it can produce noticeably better results for certain tracks.

Note that this model natively produces a single vocals stem. The instrumental track, if selected, is produced by subtracting the vocal stem from the input track:
```
instrumental = input_track - vocal_stem
```

This model was trained by [@KimberleyJensen](https://github.com/KimberleyJensen)

Take a look at the GitHub project, here: [https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)

Note that there is also a HuggingFace space here: [https://huggingface.co/KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer)

For use with [OpenVINO AI Plugins for Audacity](https://github.com/intel/openvino-plugins-ai-audacity), the pytorch source models were converted to OpenVINO IR format, and stored here: TODO

License: [GPL-3.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/gpl-3.0.md)