# Kokoro-82M

Kokoro is a lightweight, high-quality text-to-speech model with 82 million parameters. It supports multiple voices and languages.

The model ships with a large set of speaker embedding voices across several languages, including US English, British English, French, Spanish, Japanese, Chinese, Hindi, Italian, Portuguese, and Vietnamese.

Source model: [https://huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

For use with [OpenVINO AI Plugins for Audacity](https://github.com/intel/openvino-plugins-ai-audacity), the model is converted to OpenVINO IR format.

Since a public OpenVINO-converted download location is not yet available, this model cannot currently be downloaded via the Model Manager.

To use it, place the converted model folder (containing `openvino_model.xml`, `openvino_model.bin`, `config.json`, and the `voices/` and `data/` subdirectories) into `openvino-models/speech_generation/kokoro-82m/`, and restart Audacity.

After restart, the model should appear as installed/selectable in the Text-to-Speech generator and in the Model Manager.

## License

[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)