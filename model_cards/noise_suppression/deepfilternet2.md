# DeepFilterNet2

A Noise Suppression model that operates on Full-Band Audio (48kHz) mono tracks. 

Note: You can process stereo tracks as well, and in this case the model will be run for the left & right tracks, sequentially.

This is a port of the DeepFilterNet2 features from this project: [https://github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)

For use with [OpenVINO AI Plugins for Audacity](https://github.com/intel/openvino-plugins-ai-audacity), the pytorch source models were converted to OpenVINO IR format, and posted here: [https://huggingface.co/Intel/deepfilternet-openvino](https://huggingface.co/Intel/deepfilternet-openvino)

License: [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)