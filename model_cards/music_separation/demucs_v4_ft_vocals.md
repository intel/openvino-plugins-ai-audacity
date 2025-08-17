# Demucs-v4 FT Vocals

A FT (fine-tuned) variant of Demucs-v4 that gives slightly better results for vocals, as compared with the 'base' version of Demucs-v4.

Note that this model natively produces a single vocals stem. The instrumental track, if selected, is produced by subtracting the vocal stem from the input track:
```
instrumental = input_track - vocal_stem
```

For more info, take a look at the source project, here: [https://github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)

For use with [OpenVINO AI Plugins for Audacity](https://github.com/intel/openvino-plugins-ai-audacity), the pytorch models source models were converted to OpenVINO IR format, and stored here: [https://huggingface.co/Intel/demucs-openvino](https://huggingface.co/Intel/demucs-openvino)

License: [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)