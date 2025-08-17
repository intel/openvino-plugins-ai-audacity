# Demucs-v4 6s

A 6-stem variant of Demucs v4 that can separate an input track into drums, bass, vocals, guitar, piano, and 'other instruments' stems.

For more info, take a look at the source project, here: [https://github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)

*Note* that this model is considered to be *experimental*, and it is included here for completeness. To quote the *demucs* project README from [here](https://github.com/facebookresearch/demucs?tab=readme-ov-file#demucs-music-source-separation): "We are also releasing an experimental 6 sources model, that adds a `guitar` and `piano` source. Quick testing seems to show okay quality for `guitar`, but a lot of bleeding and artifacts for the `piano` source."

For use with [OpenVINO AI Plugins for Audacity](https://github.com/intel/openvino-plugins-ai-audacity), the pytorch models source models were converted to OpenVINO IR format, and stored here: [https://huggingface.co/Intel/demucs-openvino](https://huggingface.co/Intel/demucs-openvino)

License: [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)