# MDX23C Drum Separation (@jarredou)

A MDX23C-based drum sepration model that can produce 5 stems: kick, snare, toms, hi-hat, cymbals

As this is a MDX23C model, expect long processing times.

Note that this model natively produces a five stems: kick, snare, toms, hi-hat, & cymbals

There is an additional 'residual' track that is produced, which contains the 'leftover' of whatever is not contained in these five native stems.

It is produced using the following formula:
```
residual = input_track - (kick + snare + toms + hi-hat + cymbals)
```

This model was trained by [@jarredou](https://github.com/jarredou).

The original pytorch checkpoint / config were downloaded from their GitHub release here: [https://github.com/jarredou/models/releases/tag/DrumSep](https://github.com/jarredou/models/releases/tag/DrumSep)

For use with [OpenVINO AI Plugins for Audacity](https://github.com/intel/openvino-plugins-ai-audacity), the pytorch source models were converted to OpenVINO IR format, and stored here: TODO

License: [Attribution-NonCommercial-NoDerivatives 4.0 International](https://github.com/jarredou/models?tab=License-1-ov-file#readme)