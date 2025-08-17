# MelBandRoformer Crowd Extraction / Removal (@aufr33 & @viperx)

A MelBandRoformer-based crowd extraction / removal model that excels at extracting crowd noise from live recordings. As this is a MelBandRoformer model, it's quite a heavy model so expect long processing times.

Note that this model natively produces a single 'crowd' stem. The 'no crowd' is produced by subtracting the 'crowd' stem from the input track:
```
no_crowd = input_track - crowd
```

This model was trained by [@aufr33](https://github.com/aufr33) & [@viperx](https://github.com/playdasegunda)

The source pytorch models (checkpoint) & config were originally posted here: [https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v.1.0.4](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v.1.0.4)

For use with [OpenVINO AI Plugins for Audacity](https://github.com/intel/openvino-plugins-ai-audacity), the pytorch source models were converted to OpenVINO IR format, and stored here: TODO

License: [MIT](https://github.com/ZFTurbo/Music-Source-Separation-Training?tab=MIT-1-ov-file#readme)