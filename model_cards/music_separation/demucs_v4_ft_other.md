# Demucs-v4 FT Other Instruments

A FT (fine-tuned) variant of Demucs-v4 that gives slightly better results for 'other instruments', as compared with the native version of Demucs-v4.

For Demucs-v4, 'Other Instruments' are classified as anything that is *not* drums, bass, or vocals.

Note that this model natively produces a single 'other instruments' stem. The instrumental track, if selected, is produced by subtracting the 'other instruments' stem from the input track:
```
instrumental = input_track - other_instruments_stem
```

For more info, take a look at the source project, here: [https://github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)

For use with [OpenVINO AI Plugins for Audacity](https://github.com/intel/openvino-plugins-ai-audacity), the pytorch models source models were converted to OpenVINO IR format, and stored here: [https://huggingface.co/Intel/demucs-openvino](https://huggingface.co/Intel/demucs-openvino)

License: [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)