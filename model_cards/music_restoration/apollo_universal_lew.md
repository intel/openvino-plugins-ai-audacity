# Apollo Universal Restore (@Lew)

An Apollo-based lossy restoration model that works well for any lossy files.

Trained by @Lew.

The pre-trained model (by @Lew) & config was originally uploaded via Discord, and eventually reposted by @deton24 here: [https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/tag/uni](https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/tag/uni)

Unfortunately, since there is no 'official' HuggingFace source repo from @Lew with clear license available, we are unable to provide pre-converted OpenVINO models at this time.

If you're brave, and want to try converting it yourself -- You can use guidance from [here](https://github.com/RyanMetcalfeInt8/Music-Source-Separation-Training/tree/openvino_conversion/openvino_conversion#convert-apollo-models)

If you do convert it yourself, you should create a `apollo_universal` folder that contains `apollo_fwd.xml` & `apollo_fwd.bin` (output of conversion steps), and place this folder in `openvino-models/music_restoration/`, and restart Audacity. After this, the model should show up as selectable from the drop-down list.