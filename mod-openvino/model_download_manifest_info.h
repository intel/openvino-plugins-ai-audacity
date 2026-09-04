// Auto-generated header containing model download metadata
#pragma once

#include <cstddef>
#include <cstdint>

namespace model_download_manifest {

struct ModelFileInfo
{
   const char* name;
   const char* expected_sha256;
   std::uint64_t expected_size;
};

struct ModelInfo
{
   const char* effect;
   const char* model_id;
   const char* model_name;
   const char* info_key;
   const char* base_url;
   const char* revision;
   const char* post_url;
   const char* relative_path;
   const char* const* dependencies;
   std::size_t dependency_count;
   const ModelFileInfo* files;
   std::size_t file_count;
};

inline constexpr ModelFileInfo kFiles_music_restoration_apollo_mp3_jusperlee[] = {
   { "apollo_fwd.xml", "4cfc5c85eb8f5e78e8eb220724be75728ce1c703e442e4404f0117fc2f1719c1", 4473711ULL },
   { "apollo_fwd.bin", "5c1d9a6397827b698416c994ca99e1385fbcab953210264a52abea4f00fae83d", 33034848ULL },
};

inline constexpr const char* const* kDependencies_music_restoration_apollo_mp3_jusperlee = nullptr;

inline constexpr ModelFileInfo kFiles_music_restoration_apollo_universal_lew[] = {
   { "apollo_fwd.xml", "", 0ULL },
   { "apollo_fwd.bin", "", 0ULL },
};

inline constexpr const char* const* kDependencies_music_restoration_apollo_universal_lew = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4[] = {
   { "htdemucs_fwd.bin", "74c4e1ebd68b648ea5a2aabefb65d5c3ba805658037fc97a37b84ce9c72f4eb1", 104552746ULL },
   { "htdemucs_fwd.xml", "1557f96f11cf1667beca60679926bf43b393d3ff71eb4b9a625c6a07bc23f4a4", 1477599ULL },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4 = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4_ft_drums[] = {
   { "htdemucs_fwd.bin", "340e5f6d8dd4d5d0d987f545b922a8252cc5ea6e53649f91eadd8b36e77fcbac", 104552746ULL },
   { "htdemucs_fwd.xml", "1557f96f11cf1667beca60679926bf43b393d3ff71eb4b9a625c6a07bc23f4a4", 1477599ULL },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4_ft_drums = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4_ft_bass[] = {
   { "htdemucs_fwd.bin", "a7ec34d838b0f8f36b11a90d6f4e588141a383faadc77e9648e19fdd9eefc95e", 104552746ULL },
   { "htdemucs_fwd.xml", "1557f96f11cf1667beca60679926bf43b393d3ff71eb4b9a625c6a07bc23f4a4", 1477599ULL },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4_ft_bass = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4_ft_other[] = {
   { "htdemucs_fwd.bin", "c5f4bb99e91fea0bda75f64086712884457a5c9f861d1c2a7a02e09e92c2846e", 104552746ULL },
   { "htdemucs_fwd.xml", "1557f96f11cf1667beca60679926bf43b393d3ff71eb4b9a625c6a07bc23f4a4", 1477599ULL },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4_ft_other = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4_ft_vocals[] = {
   { "htdemucs_fwd.bin", "8576b4bd27095bc7f2dae39c529124b04eebc3e04009a45b7cc170f5b563f749", 104552746ULL },
   { "htdemucs_fwd.xml", "1557f96f11cf1667beca60679926bf43b393d3ff71eb4b9a625c6a07bc23f4a4", 1477599ULL },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4_ft_vocals = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4_6s[] = {
   { "htdemucs_fwd.bin", "5d66281099b5fdaba8844e306f3d6710b3e0b005fa9aad4514ce7604e24fbec9", 74384618ULL },
   { "htdemucs_fwd.xml", "06c68a8a21a6cefb905575cbe689cfa3241416a4c263271804ae5690f8ba5cbf", 1461204ULL },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4_6s = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_mel_vocals_kimberley_jenson[] = {
   { "mel_band_pre.xml", "863afb676b3401a0030b3e84aa34db436d553491bf490b2e3f2ea04fea2a03f0", 25802ULL },
   { "mel_band_pre.bin", "ded35e388bb5662a161003d4036534a03cd7efa45e3c0c1383f1f1acb73d7f9a", 4242ULL },
   { "mel_band_post.xml", "b87793f6eedb7cb66d0e6c9efbb23b130ad494f360689ef32a5c19afd906320e", 49300ULL },
   { "mel_band_post.bin", "b7ebaa78cc9468995474b9a05c9887079fb806f4c861c834b1e9161effde447c", 36094ULL },
   { "mel_band_fwd.xml", "e9f681f7184f542338e3c963ba837fd2e622e08bf27e6592aeae92fca757efa4", 2470791ULL },
   { "mel_band_fwd.bin", "1a23e353c7b41267ef683beb9be2d1c5f25a7cc2725b43f6beee68cb7cfc1a0b", 456659406ULL },
};

inline constexpr const char* const* kDependencies_music_separation_mel_vocals_kimberley_jenson = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_mel_crowd_aufr33_viperx[] = {
   { "mel_band_pre.xml", "863afb676b3401a0030b3e84aa34db436d553491bf490b2e3f2ea04fea2a03f0", 25802ULL },
   { "mel_band_pre.bin", "ded35e388bb5662a161003d4036534a03cd7efa45e3c0c1383f1f1acb73d7f9a", 4242ULL },
   { "mel_band_post.xml", "b87793f6eedb7cb66d0e6c9efbb23b130ad494f360689ef32a5c19afd906320e", 49300ULL },
   { "mel_band_post.bin", "b7ebaa78cc9468995474b9a05c9887079fb806f4c861c834b1e9161effde447c", 36094ULL },
   { "mel_band_fwd.xml", "e9f681f7184f542338e3c963ba837fd2e622e08bf27e6592aeae92fca757efa4", 2470791ULL },
   { "mel_band_fwd.bin", "bf585466ba9b3666d16ad18e0ca40cba59c6bf67a7017a95014d81f8aec0317a", 456659406ULL },
};

inline constexpr const char* const* kDependencies_music_separation_mel_crowd_aufr33_viperx = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_msdx23c_drum_sep_jarredou[] = {
   { "mdx23c_fwd.xml", "62c7107ad49bac24bb4e0839971387d6053446c6aa234cf99bce7af7f6ec532c", 775186ULL },
   { "mdx23c_fwd.bin", "682f8616845a9f31bb21df38198df2d36ef79e96182d0ac7671d332afc80270e", 218753348ULL },
};

inline constexpr const char* const* kDependencies_music_separation_msdx23c_drum_sep_jarredou = nullptr;

inline constexpr ModelFileInfo kFiles_noise_suppression_deepfilternet2[] = {
   { "df_dec.bin", "19862c41d157cceb1fd048087709f3a9b187fab883677ba12bf538f1bc5cfd14", 3321828ULL },
   { "df_dec.xml", "cde83f4a1b6dbc017b2966dc0bd6c389360a90988ef99a00639e7e8c9a60a361", 110816ULL },
   { "enc.xml", "667881baeef4e44b66642e1ba4fed267b02a818ada05f57efecfc9cc755f2e07", 173466ULL },
   { "enc.bin", "ed35e70c2e8c8035909466c82af4fd447238821734bc2d85c92efe721a49145a", 2589012ULL },
   { "erb_dec.xml", "d06b04a3ece3079b3fd1869f9960a7c3c9db6cc724162c0efc2dadc6b40d8a81", 178896ULL },
   { "erb_dec.bin", "f1df0fb3173ef334abf338b1f11e55c40ec413abdd77cfbf80867294c3de937f", 3311612ULL },
};

inline constexpr const char* const* kDependencies_noise_suppression_deepfilternet2 = nullptr;

inline constexpr ModelFileInfo kFiles_noise_suppression_deepfilternet3[] = {
   { "df_dec.bin", "7bd50b92ae1c6175be9ddb3e8cf7f81b65b95223957bcd558cb2c6ef281aa435", 3325940ULL },
   { "df_dec.xml", "7366c6715fb2c0febbcd05b0d3f1e47d48eaeeb213fb2d7eeb2f4aa24b63dc42", 122326ULL },
   { "enc.xml", "ed385d86a59d84fcebf11bdceb2b2e3d2a7b4a1379a744dc9b0d47121622df32", 184737ULL },
   { "enc.bin", "fa1ed5feb4dd234e051188a2cda6c6f6efabefd87404cbae20fe6c9913dc9838", 1934676ULL },
   { "erb_dec.xml", "bc0cf4e25911714e074658cfeeccc686787a4d274889a88afef855b542689faa", 182978ULL },
   { "erb_dec.bin", "d5e9dc11c071f72b343d5af5d344a504a903347d455952e8d4027da3ab47e792", 3278844ULL },
};

inline constexpr const char* const* kDependencies_noise_suppression_deepfilternet3 = nullptr;

inline constexpr ModelFileInfo kFiles_noise_suppression_denseunet[] = {
   { "noise-suppression-denseunet-ll-0001.xml", "89116a01cc59f7ac3f1f1365c851e47c9089fd33ca86712c30dc1f0ee7d14803", 689820ULL },
   { "noise-suppression-denseunet-ll-0001.bin", "da59b41a656b2948a4b45580b4870614d2dfa071a7566555aea4547423888e08", 8625568ULL },
};

inline constexpr const char* const* kDependencies_noise_suppression_denseunet = nullptr;

inline constexpr ModelFileInfo kFiles_reverb_removal_mel_band_roformer_mono_anvuew[] = {
   { "mel_band_pre.xml", "863afb676b3401a0030b3e84aa34db436d553491bf490b2e3f2ea04fea2a03f0", 25802ULL },
   { "mel_band_pre.bin", "ded35e388bb5662a161003d4036534a03cd7efa45e3c0c1383f1f1acb73d7f9a", 4242ULL },
   { "mel_band_post.xml", "b87793f6eedb7cb66d0e6c9efbb23b130ad494f360689ef32a5c19afd906320e", 49300ULL },
   { "mel_band_post.bin", "b7ebaa78cc9468995474b9a05c9887079fb806f4c861c834b1e9161effde447c", 36094ULL },
   { "mel_band_fwd.xml", "e9f681f7184f542338e3c963ba837fd2e622e08bf27e6592aeae92fca757efa4", 2470791ULL },
   { "mel_band_fwd.bin", "aacebaac7db6756c8e954331610f4664ad3ff1212df50feda6c4446b7d1155e8", 456659406ULL },
};

inline constexpr const char* const* kDependencies_reverb_removal_mel_band_roformer_mono_anvuew = nullptr;

inline constexpr ModelFileInfo kFiles_super_resolution_common[] = {
   { "audiosr_decoder.bin", "3f15dd621d5ffb191f1dbb5540b7c75301e8769ccbe062942b74a81e98ac2dd2", 267047980ULL },
   { "audiosr_decoder.xml", "bc650ad5ce4e03db47640e2a777b78984d6d078a8cf0a13f166ea3ac14209016", 247413ULL },
   { "audiosr_encoder.bin", "9eb70d3baff6a5859a8f05f6ab7b3a13d5d0b23540758c78ccf91b0ccd4f83d6", 180020602ULL },
   { "audiosr_encoder.xml", "68f54ed5d3bf86c0be87419a70328470231bf4dfba616d7c44817591418a5095", 268416ULL },
   { "mel_24000_cpu.raw", "8dd24b6b1a81dc8f70ed479186fe66b1ccb4fc2dcffc3133dfd01daa93618c1d", 1049600ULL },
   { "post_quant_conv.bin", "fe865e2ad452ac5de682a7d9afecd685da2fca328ff453d0575fd75196f809c0", 544ULL },
   { "post_quant_conv.xml", "35437f666d07fda6c61c4d226111e26b99d58801aa3b7a7a941e9e414c8b862b", 4127ULL },
   { "quant_conv.bin", "5a10eb725d47b58dd4edf72284681c78edcd56c83386e2665d3f7dcdf22ff96f", 2112ULL },
   { "quant_conv.xml", "9a481b5cb353c050da3377bc709a9f939d41533221810a1ee23eea474f2a93c4", 4134ULL },
   { "vae_feature_extract.bin", "9cb961e646544a8de08cddef328aabe222e7e701794bb37193cd651f522a188f", 180022744ULL },
   { "vae_feature_extract.xml", "307cd9825f64f72da24a5737df620b76443e5effbeaf34637ca1f7c1f6e60599", 296881ULL },
   { "vocoder.xml", "387b3a16868146e78d321eb58abfcaa2ec4ecac80ab1f1196a00260b94b99e67", 592981ULL },
   { "vocoder.bin", "33a9c6375fe33bc652f8e22d4fc0b5317351f9acdaa7121bf92c9b1e594fdd8f", 380563784ULL },
};

inline constexpr const char* const* kDependencies_super_resolution_common = nullptr;

inline constexpr ModelFileInfo kFiles_super_resolution_basic_general_fp16[] = {
   { "basic/ddpm.xml", "70915c83faf9e66df638328d00c391daf3aefb68e66aa916022e13ddb5f51c0e", 4355984ULL },
   { "basic/ddpm.bin", "03fa2b86f718e23b0c81e1ef481c1d75c87d3d6eb949cdfe06ca351b8f509d8d", 516390728ULL },
};

inline constexpr const char* kDependencies_super_resolution_basic_general_fp16[] = {
   "super_resolution_common",
};

inline constexpr ModelFileInfo kFiles_super_resolution_speech_fp16[] = {
   { "speech/ddpm.xml", "70915c83faf9e66df638328d00c391daf3aefb68e66aa916022e13ddb5f51c0e", 4355984ULL },
   { "speech/ddpm.bin", "3565cdfca685b7d6bbce89257dd62c54827bb5b5452ce5bf5439e271fa2d2c8d", 516390728ULL },
};

inline constexpr const char* kDependencies_super_resolution_speech_fp16[] = {
   "super_resolution_common",
};

inline constexpr ModelFileInfo kFiles_text_to_speech_kokoro_82m[] = {
   { "openvino_model.xml", "", 0ULL },
   { "openvino_model.bin", "", 0ULL },
   { "config.json", "", 0ULL },
   { "data/gb_gold.json", "", 0ULL },
   { "data/gb_silver.json", "", 0ULL },
   { "data/ja_words.txt", "", 0ULL },
   { "data/us_gold.json", "", 0ULL },
   { "data/us_silver.json", "", 0ULL },
   { "data/vi_acronyms.json", "", 0ULL },
   { "data/vi_symbols.json", "", 0ULL },
   { "data/vi_teencode.json", "", 0ULL },
   { "voices/af_alloy.bin", "", 0ULL },
   { "voices/af_aoede.bin", "", 0ULL },
   { "voices/af_bella.bin", "", 0ULL },
   { "voices/af_heart.bin", "", 0ULL },
   { "voices/af_jessica.bin", "", 0ULL },
   { "voices/af_kore.bin", "", 0ULL },
   { "voices/af_nicole.bin", "", 0ULL },
   { "voices/af_nova.bin", "", 0ULL },
   { "voices/af_river.bin", "", 0ULL },
   { "voices/af_sarah.bin", "", 0ULL },
   { "voices/af_sky.bin", "", 0ULL },
   { "voices/am_adam.bin", "", 0ULL },
   { "voices/am_echo.bin", "", 0ULL },
   { "voices/am_eric.bin", "", 0ULL },
   { "voices/am_fenrir.bin", "", 0ULL },
   { "voices/am_liam.bin", "", 0ULL },
   { "voices/am_michael.bin", "", 0ULL },
   { "voices/am_onyx.bin", "", 0ULL },
   { "voices/am_puck.bin", "", 0ULL },
   { "voices/am_santa.bin", "", 0ULL },
   { "voices/bf_alice.bin", "", 0ULL },
   { "voices/bf_emma.bin", "", 0ULL },
   { "voices/bf_isabella.bin", "", 0ULL },
   { "voices/bf_lily.bin", "", 0ULL },
   { "voices/bm_daniel.bin", "", 0ULL },
   { "voices/bm_fable.bin", "", 0ULL },
   { "voices/bm_george.bin", "", 0ULL },
   { "voices/bm_lewis.bin", "", 0ULL },
   { "voices/ef_dora.bin", "", 0ULL },
   { "voices/em_alex.bin", "", 0ULL },
   { "voices/em_santa.bin", "", 0ULL },
   { "voices/ff_siwis.bin", "", 0ULL },
   { "voices/hf_alpha.bin", "", 0ULL },
   { "voices/hf_beta.bin", "", 0ULL },
   { "voices/hm_omega.bin", "", 0ULL },
   { "voices/hm_psi.bin", "", 0ULL },
   { "voices/if_sara.bin", "", 0ULL },
   { "voices/im_nicola.bin", "", 0ULL },
   { "voices/jf_alpha.bin", "", 0ULL },
   { "voices/jf_gongitsune.bin", "", 0ULL },
   { "voices/jf_nezumi.bin", "", 0ULL },
   { "voices/jf_tebukuro.bin", "", 0ULL },
   { "voices/jm_kumo.bin", "", 0ULL },
   { "voices/pf_dora.bin", "", 0ULL },
   { "voices/pm_alex.bin", "", 0ULL },
   { "voices/pm_santa.bin", "", 0ULL },
   { "voices/zf_xiaobei.bin", "", 0ULL },
   { "voices/zf_xiaoni.bin", "", 0ULL },
   { "voices/zf_xiaoxiao.bin", "", 0ULL },
   { "voices/zf_xiaoyi.bin", "", 0ULL },
   { "voices/zm_yunjian.bin", "", 0ULL },
   { "voices/zm_yunxi.bin", "", 0ULL },
   { "voices/zm_yunxia.bin", "", 0ULL },
   { "voices/zm_yunyang.bin", "", 0ULL },
};

inline constexpr const char* const* kDependencies_text_to_speech_kokoro_82m = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_base_fp16[] = {
   { "added_tokens.json", "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1", 34604ULL },
   { "config.json", "7580357cd33e60d3c453b90b6ab4d78606cbc5de32cc2da8df873468dcfac237", 1320ULL },
   { "generation_config.json", "a24ccb25a8638fb23d3acfa0b234982497dd30bb3c577d145619525629f38e97", 3802ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "36f67da0ba9327bf817a258855bf696ad01cbf899268bc6c06f7e10c448f0427", 104006812ULL },
   { "openvino_decoder_model.xml", "272e09a043e5f08bb8d9f0c93f28d3e33aa08aac065be6a4be397c9e4c8bae9d", 496004ULL },
   { "openvino_detokenizer.bin", "2542d1fe6c4c5d838e7b7c61b24996b94ae0cb54e64ff6dbaab2b9dd5ddd7ca0", 736181ULL },
   { "openvino_detokenizer.xml", "ccb191d7e6bd5d4f5f5fcf3ffac54a6c109d60280f4befc1f7996625db594ead", 9699ULL },
   { "openvino_encoder_model.bin", "9ee2fa771d004d1af47ffc7da0909f2025d43b7df05e5c92532d959b9c2db234", 41181290ULL },
   { "openvino_encoder_model.xml", "5920849e30a3af25bcb674c539862d418e1b7ba4fe666105fb66cfd97e4d450d", 241299ULL },
   { "openvino_tokenizer.bin", "846f3c65f7a71f120fce7aaaf41b342f0767eb13e3c4e7c2a54f1d49b5c38fda", 1898933ULL },
   { "openvino_tokenizer.xml", "01051cc5271b954a660dbe6a8a72c8266c65461678a62ab478fac138942bc411", 27011ULL },
   { "preprocessor_config.json", "994838f1fa6462c8b9b3c90edada831f11f3dd8b4664634e18f4694d005c9dbf", 356ULL },
   { "special_tokens_map.json", "e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691", 2194ULL },
   { "tokenizer.json", "7b469ff15eb7816315aa45eec391f5943d639b9d73d110f5c003df5192fd54e3", 3930494ULL },
   { "tokenizer_config.json", "21a4fc0483c14b87f4e0bbc177a9a357479bfa7c95aaf21ad53d71e9c5afafb9", 282713ULL },
   { "vocab.json", "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9", 835550ULL },
};

inline constexpr const char* const* kDependencies_whisper_base_fp16 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_base_int8[] = {
   { "added_tokens.json", "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1", 34604ULL },
   { "config.json", "7580357cd33e60d3c453b90b6ab4d78606cbc5de32cc2da8df873468dcfac237", 1320ULL },
   { "generation_config.json", "a24ccb25a8638fb23d3acfa0b234982497dd30bb3c577d145619525629f38e97", 3802ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "1e373d629c4a1a5c7a1964188f77c2b52b0bb488f27015fd9dc35bcb1e7eef42", 52439987ULL },
   { "openvino_decoder_model.xml", "901c8584bd9c7721b879d617d2941fb9c318150cfec5d04056bc8faaa9745285", 564428ULL },
   { "openvino_detokenizer.bin", "2542d1fe6c4c5d838e7b7c61b24996b94ae0cb54e64ff6dbaab2b9dd5ddd7ca0", 736181ULL },
   { "openvino_detokenizer.xml", "2ef46e0d325a858753784b4f96172bd0e627cd9ff1fd0c9db5ac8bc1785b8c7f", 9699ULL },
   { "openvino_encoder_model.bin", "a0aa4518850411dfadc7799b426d7c08e966a85367be96588432fbadf40789d8", 23097456ULL },
   { "openvino_encoder_model.xml", "79dc09241718475ca14277bb16766cfb688b279f412cae8972b1c1857863ae3a", 295834ULL },
   { "openvino_tokenizer.bin", "846f3c65f7a71f120fce7aaaf41b342f0767eb13e3c4e7c2a54f1d49b5c38fda", 1898933ULL },
   { "openvino_tokenizer.xml", "3e4ddd6e2031c307d5db367630298ae77d6f9f5675b689cb536e0da617a0e38d", 27011ULL },
   { "preprocessor_config.json", "994838f1fa6462c8b9b3c90edada831f11f3dd8b4664634e18f4694d005c9dbf", 356ULL },
   { "special_tokens_map.json", "e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691", 2194ULL },
   { "tokenizer.json", "7b469ff15eb7816315aa45eec391f5943d639b9d73d110f5c003df5192fd54e3", 3930494ULL },
   { "tokenizer_config.json", "21a4fc0483c14b87f4e0bbc177a9a357479bfa7c95aaf21ad53d71e9c5afafb9", 282713ULL },
   { "vocab.json", "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9", 835550ULL },
};

inline constexpr const char* const* kDependencies_whisper_base_int8 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_base_int4[] = {
   { "added_tokens.json", "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1", 34604ULL },
   { "config.json", "7580357cd33e60d3c453b90b6ab4d78606cbc5de32cc2da8df873468dcfac237", 1320ULL },
   { "generation_config.json", "a24ccb25a8638fb23d3acfa0b234982497dd30bb3c577d145619525629f38e97", 3802ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "2dd4497dc017a79fd7816355a90eae0578041acb0db09d706f71a6367cdb4999", 40228835ULL },
   { "openvino_decoder_model.xml", "8002db18f58e105aa6978f2ab0954bef39e7dd89799143a9eb96b48d77c6f898", 627401ULL },
   { "openvino_detokenizer.bin", "2542d1fe6c4c5d838e7b7c61b24996b94ae0cb54e64ff6dbaab2b9dd5ddd7ca0", 736181ULL },
   { "openvino_detokenizer.xml", "2ef46e0d325a858753784b4f96172bd0e627cd9ff1fd0c9db5ac8bc1785b8c7f", 9699ULL },
   { "openvino_encoder_model.bin", "1e27d9ae27adaf114c8ec1736d8b9afcc6f73702984736d43cc6678a4dcfaf37", 14451360ULL },
   { "openvino_encoder_model.xml", "9dffc149d8726bdbb980c45f191a61bc8cc3e1beb9729dbf6c92bd326b55e017", 332326ULL },
   { "openvino_tokenizer.bin", "846f3c65f7a71f120fce7aaaf41b342f0767eb13e3c4e7c2a54f1d49b5c38fda", 1898933ULL },
   { "openvino_tokenizer.xml", "3e4ddd6e2031c307d5db367630298ae77d6f9f5675b689cb536e0da617a0e38d", 27011ULL },
   { "preprocessor_config.json", "994838f1fa6462c8b9b3c90edada831f11f3dd8b4664634e18f4694d005c9dbf", 356ULL },
   { "special_tokens_map.json", "e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691", 2194ULL },
   { "tokenizer.json", "7b469ff15eb7816315aa45eec391f5943d639b9d73d110f5c003df5192fd54e3", 3930494ULL },
   { "tokenizer_config.json", "21a4fc0483c14b87f4e0bbc177a9a357479bfa7c95aaf21ad53d71e9c5afafb9", 282713ULL },
   { "vocab.json", "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9", 835550ULL },
};

inline constexpr const char* const* kDependencies_whisper_base_int4 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_medium_fp16[] = {
   { "added_tokens.json", "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1", 34604ULL },
   { "config.json", "468a4c8675c49b8ab27717fe02223b6ae2565dc2b6411d017e7e04aec8497dec", 1326ULL },
   { "generation_config.json", "d5dd0ffb1125f44ecae80a93f04c06c8127c2a2aff4ea2feec171d2fda919612", 3750ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "d89a0b3d8a77cb4b619a3555db9fdee5bf82b2254ff0a50ae8dc9edc1d0aec25", 913283228ULL },
   { "openvino_decoder_model.xml", "8fb57e209b791338b46e3f5b63e572ab216c7e10ad3141071d984b3ce470ab8f", 1893862ULL },
   { "openvino_detokenizer.bin", "2542d1fe6c4c5d838e7b7c61b24996b94ae0cb54e64ff6dbaab2b9dd5ddd7ca0", 736181ULL },
   { "openvino_detokenizer.xml", "3789bfaca5c267e67967bd7429b370f0ee9c65c03395aa091b0b59fc14f475ba", 9699ULL },
   { "openvino_encoder_model.bin", "8784a18663a2794dfe89624e6f775de29fe24546acffc4da4fd8bb2b0173cf33", 614432874ULL },
   { "openvino_encoder_model.xml", "9a539021be2a7e39aa7a0a0fb58f74356200220bc0d78580ed0d7eaf8c85ffc8", 931771ULL },
   { "openvino_tokenizer.bin", "846f3c65f7a71f120fce7aaaf41b342f0767eb13e3c4e7c2a54f1d49b5c38fda", 1898933ULL },
   { "openvino_tokenizer.xml", "fe367cf66aa051a81dffa6e0b2865a201a5e603df6c527f14457b99a30981459", 27011ULL },
   { "preprocessor_config.json", "994838f1fa6462c8b9b3c90edada831f11f3dd8b4664634e18f4694d005c9dbf", 356ULL },
   { "special_tokens_map.json", "e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691", 2194ULL },
   { "tokenizer.json", "7b469ff15eb7816315aa45eec391f5943d639b9d73d110f5c003df5192fd54e3", 3930494ULL },
   { "tokenizer_config.json", "21a4fc0483c14b87f4e0bbc177a9a357479bfa7c95aaf21ad53d71e9c5afafb9", 282713ULL },
   { "vocab.json", "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9", 835550ULL },
};

inline constexpr const char* const* kDependencies_whisper_medium_fp16 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_medium_int8[] = {
   { "added_tokens.json", "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1", 34604ULL },
   { "config.json", "468a4c8675c49b8ab27717fe02223b6ae2565dc2b6411d017e7e04aec8497dec", 1326ULL },
   { "generation_config.json", "d5dd0ffb1125f44ecae80a93f04c06c8127c2a2aff4ea2feec171d2fda919612", 3750ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "4ff8ba47640c78e30fdfd6b1c91f69ead9f67047877ee38baf83349927b429eb", 459016627ULL },
   { "openvino_decoder_model.xml", "ddc857598959ed2b6f290172da1478cfa29249f1b5b8080ed91e8048e266f7fa", 2160825ULL },
   { "openvino_detokenizer.bin", "2542d1fe6c4c5d838e7b7c61b24996b94ae0cb54e64ff6dbaab2b9dd5ddd7ca0", 736181ULL },
   { "openvino_detokenizer.xml", "9adb820ef4576d3c317bf897afa04255f1fe1f87f63650e60143894e6ff6f2ba", 9699ULL },
   { "openvino_encoder_model.bin", "e74ff119334068b54a6da0b039d9f8a7a8caa5a2b0a5e940f11eb4a20b2251e1", 313391216ULL },
   { "openvino_encoder_model.xml", "fbd9387fee6a1da4ebb779227aeb77f4520370f388c0af8e215f0ad1a7cb2a44", 1142319ULL },
   { "openvino_tokenizer.bin", "846f3c65f7a71f120fce7aaaf41b342f0767eb13e3c4e7c2a54f1d49b5c38fda", 1898933ULL },
   { "openvino_tokenizer.xml", "f15e4c9573899623a9dee76f43ec204901d7bb2721f1e21531b3e82c63fc6362", 27011ULL },
   { "preprocessor_config.json", "994838f1fa6462c8b9b3c90edada831f11f3dd8b4664634e18f4694d005c9dbf", 356ULL },
   { "special_tokens_map.json", "e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691", 2194ULL },
   { "tokenizer.json", "7b469ff15eb7816315aa45eec391f5943d639b9d73d110f5c003df5192fd54e3", 3930494ULL },
   { "tokenizer_config.json", "21a4fc0483c14b87f4e0bbc177a9a357479bfa7c95aaf21ad53d71e9c5afafb9", 282713ULL },
   { "vocab.json", "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9", 835550ULL },
};

inline constexpr const char* const* kDependencies_whisper_medium_int8 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_medium_int4[] = {
   { "added_tokens.json", "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1", 34604ULL },
   { "config.json", "468a4c8675c49b8ab27717fe02223b6ae2565dc2b6411d017e7e04aec8497dec", 1326ULL },
   { "generation_config.json", "d5dd0ffb1125f44ecae80a93f04c06c8127c2a2aff4ea2feec171d2fda919612", 3750ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "a6330bbdc720f1b0a2c757f5fd14fbfa6844719e324d7b66686b65b8e155d19f", 264595939ULL },
   { "openvino_decoder_model.xml", "804d5c5c517a4670b5818f79ced65a3d6b2fc591ae9410b4c3fc5d823fdd93fa", 2411268ULL },
   { "openvino_detokenizer.bin", "2542d1fe6c4c5d838e7b7c61b24996b94ae0cb54e64ff6dbaab2b9dd5ddd7ca0", 736181ULL },
   { "openvino_detokenizer.xml", "9adb820ef4576d3c317bf897afa04255f1fe1f87f63650e60143894e6ff6f2ba", 9699ULL },
   { "openvino_encoder_model.bin", "05af9a9375adbc8e2c50fe8388e90ecd8d220d26380c8844201517fbd4993ffd", 169649312ULL },
   { "openvino_encoder_model.xml", "8a61644577aa080c4f5e6205d4266edf01f99567533d814789f52b69c47c9b1c", 1291921ULL },
   { "openvino_tokenizer.bin", "846f3c65f7a71f120fce7aaaf41b342f0767eb13e3c4e7c2a54f1d49b5c38fda", 1898933ULL },
   { "openvino_tokenizer.xml", "f15e4c9573899623a9dee76f43ec204901d7bb2721f1e21531b3e82c63fc6362", 27011ULL },
   { "preprocessor_config.json", "994838f1fa6462c8b9b3c90edada831f11f3dd8b4664634e18f4694d005c9dbf", 356ULL },
   { "special_tokens_map.json", "e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691", 2194ULL },
   { "tokenizer.json", "7b469ff15eb7816315aa45eec391f5943d639b9d73d110f5c003df5192fd54e3", 3930494ULL },
   { "tokenizer_config.json", "21a4fc0483c14b87f4e0bbc177a9a357479bfa7c95aaf21ad53d71e9c5afafb9", 282713ULL },
   { "vocab.json", "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9", 835550ULL },
};

inline constexpr const char* const* kDependencies_whisper_medium_int4 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_large_v2_fp16[] = {
   { "added_tokens.json", "", 0ULL },
   { "config.json", "", 0ULL },
   { "generation_config.json", "", 0ULL },
   { "normalizer.json", "", 0ULL },
   { "openvino_decoder_model.bin", "", 0ULL },
   { "openvino_decoder_model.xml", "", 0ULL },
   { "openvino_detokenizer.bin", "", 0ULL },
   { "openvino_detokenizer.xml", "", 0ULL },
   { "openvino_encoder_model.bin", "", 0ULL },
   { "openvino_encoder_model.xml", "", 0ULL },
   { "openvino_tokenizer.bin", "", 0ULL },
   { "openvino_tokenizer.xml", "", 0ULL },
   { "preprocessor_config.json", "", 0ULL },
   { "special_tokens_map.json", "", 0ULL },
   { "tokenizer.json", "", 0ULL },
   { "tokenizer_config.json", "", 0ULL },
   { "vocab.json", "", 0ULL },
};

inline constexpr const char* const* kDependencies_whisper_large_v2_fp16 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_large_v2_int8[] = {
   { "added_tokens.json", "", 0ULL },
   { "config.json", "", 0ULL },
   { "generation_config.json", "", 0ULL },
   { "normalizer.json", "", 0ULL },
   { "openvino_decoder_model.bin", "", 0ULL },
   { "openvino_decoder_model.xml", "", 0ULL },
   { "openvino_detokenizer.bin", "", 0ULL },
   { "openvino_detokenizer.xml", "", 0ULL },
   { "openvino_encoder_model.bin", "", 0ULL },
   { "openvino_encoder_model.xml", "", 0ULL },
   { "openvino_tokenizer.bin", "", 0ULL },
   { "openvino_tokenizer.xml", "", 0ULL },
   { "preprocessor_config.json", "", 0ULL },
   { "special_tokens_map.json", "", 0ULL },
   { "tokenizer.json", "", 0ULL },
   { "tokenizer_config.json", "", 0ULL },
   { "vocab.json", "", 0ULL },
};

inline constexpr const char* const* kDependencies_whisper_large_v2_int8 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_large_v2_int4[] = {
   { "added_tokens.json", "", 0ULL },
   { "config.json", "", 0ULL },
   { "generation_config.json", "", 0ULL },
   { "normalizer.json", "", 0ULL },
   { "openvino_decoder_model.bin", "", 0ULL },
   { "openvino_decoder_model.xml", "", 0ULL },
   { "openvino_detokenizer.bin", "", 0ULL },
   { "openvino_detokenizer.xml", "", 0ULL },
   { "openvino_encoder_model.bin", "", 0ULL },
   { "openvino_encoder_model.xml", "", 0ULL },
   { "openvino_tokenizer.bin", "", 0ULL },
   { "openvino_tokenizer.xml", "", 0ULL },
   { "preprocessor_config.json", "", 0ULL },
   { "special_tokens_map.json", "", 0ULL },
   { "tokenizer.json", "", 0ULL },
   { "tokenizer_config.json", "", 0ULL },
   { "vocab.json", "", 0ULL },
};

inline constexpr const char* const* kDependencies_whisper_large_v2_int4 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_large_v3_fp16[] = {
   { "added_tokens.json", "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0", 34648ULL },
   { "config.json", "2efba1fd9d65450929e2d917a32cf9dec2fd5682d29baa6ff48d31d3a61fb501", 1195ULL },
   { "generation_config.json", "041bb1619064440f4cb37a53326f5394de3df380f52ca1e2bfae599535414a7b", 3898ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "13542f8948ac82e3ffa1bacb54e38de784f35383bb2cda495ddd53e9f864577e", 1813043356ULL },
   { "openvino_decoder_model.xml", "301f769e85b11f28f0b742c4a2ad92ba59ab6267df6b8ebfc5a29ffa8ce49055", 2514921ULL },
   { "openvino_detokenizer.bin", "f2b3c47825a1089525ff65c0c8e49271e1dee69a401a04fc827ac2de5b7766e4", 736198ULL },
   { "openvino_detokenizer.xml", "55ba846c7ab78518b1e8f21923ef0c42fc78e86a0a5bd0b3f488a2e29a236775", 9699ULL },
   { "openvino_encoder_model.bin", "76de5b22435747d7f0d7f74b003acd23ab5c10b7e1198d36373cdf0f6975b630", 1273938026ULL },
   { "openvino_encoder_model.xml", "d913c09a5c5c1f86e6b5dced2a44df0b69b9aacf8218451d9100ee4fa12dec3d", 1239276ULL },
   { "openvino_tokenizer.bin", "adfa3d9a2920d0f314121270a960ab331ec0f05838544bb8ecaaa422282a6fd4", 1898973ULL },
   { "openvino_tokenizer.xml", "11ae95e07936a776ef7e547160acdce17e1b05d6b9e4da623f253120903272d6", 27011ULL },
   { "preprocessor_config.json", "654cf18d3e163b948ceaf9766da56ce0b52de265d58673cf61c9376f126bd499", 357ULL },
   { "special_tokens_map.json", "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74", 2186ULL },
   { "tokenizer.json", "5c1bf30c9e716e1477bedef846b01be0013daecb89e9e3ef7ab89b23c178df1b", 3930645ULL },
   { "tokenizer_config.json", "3c75940dfce3a294fca7041a5faff011677f1b68fa85e47511bb8cf6dccaded6", 282873ULL },
   { "vocab.json", "6788c80b082e9b0d1393147d3a3e62ba19285ac0c82ace8e5ef00f37ead58971", 835528ULL },
};

inline constexpr const char* const* kDependencies_whisper_large_v3_fp16 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_large_v3_int8[] = {
   { "added_tokens.json", "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0", 34648ULL },
   { "config.json", "2efba1fd9d65450929e2d917a32cf9dec2fd5682d29baa6ff48d31d3a61fb501", 1195ULL },
   { "generation_config.json", "041bb1619064440f4cb37a53326f5394de3df380f52ca1e2bfae599535414a7b", 3898ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "fdf13685d5a9c427b9aa5893c2baef362f1e4dddfbf5bf8a47fc03acb35a45ea", 910372790ULL },
   { "openvino_decoder_model.xml", "cc4b29bb1fc4afb6dcc59db2cf3d149fe5847a7215ca66b4944cb36390b17805", 2869528ULL },
   { "openvino_detokenizer.bin", "f2b3c47825a1089525ff65c0c8e49271e1dee69a401a04fc827ac2de5b7766e4", 736198ULL },
   { "openvino_detokenizer.xml", "ac37fdd9b044a04130747c6773b85f395e5910e0cfb132a2c937fb533d69b4da", 9699ULL },
   { "openvino_encoder_model.bin", "fffcbf47a4cfd5a1e3f57c0569f5ef706245b798ef626db5ffea7b84166ed865", 645332592ULL },
   { "openvino_encoder_model.xml", "2a19e0ae9b77b5fbf2928793aa232eb84b0554a29fd4f16e190617583bdefcfc", 1518681ULL },
   { "openvino_tokenizer.bin", "adfa3d9a2920d0f314121270a960ab331ec0f05838544bb8ecaaa422282a6fd4", 1898973ULL },
   { "openvino_tokenizer.xml", "dad2092c08510c043acabeae952cdbd381bc7b9c3272eaef524fa4fbea2f9dc4", 27011ULL },
   { "preprocessor_config.json", "654cf18d3e163b948ceaf9766da56ce0b52de265d58673cf61c9376f126bd499", 357ULL },
   { "special_tokens_map.json", "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74", 2186ULL },
   { "tokenizer.json", "5c1bf30c9e716e1477bedef846b01be0013daecb89e9e3ef7ab89b23c178df1b", 3930645ULL },
   { "tokenizer_config.json", "3c75940dfce3a294fca7041a5faff011677f1b68fa85e47511bb8cf6dccaded6", 282873ULL },
   { "vocab.json", "6788c80b082e9b0d1393147d3a3e62ba19285ac0c82ace8e5ef00f37ead58971", 835528ULL },
};

inline constexpr const char* const* kDependencies_whisper_large_v3_int8 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_large_v3_int4[] = {
   { "added_tokens.json", "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0", 34648ULL },
   { "config.json", "2efba1fd9d65450929e2d917a32cf9dec2fd5682d29baa6ff48d31d3a61fb501", 1195ULL },
   { "generation_config.json", "041bb1619064440f4cb37a53326f5394de3df380f52ca1e2bfae599535414a7b", 3898ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "9b6c32730575fd4792dfa9c01dc432033ccd6f4c5819cbfee29885f12dfe9484", 505728998ULL },
   { "openvino_decoder_model.xml", "a3f84c5e34a732695d4ed09faddfa639a3a6d05de1eff90d1e2f7167f410f99f", 3208882ULL },
   { "openvino_detokenizer.bin", "f2b3c47825a1089525ff65c0c8e49271e1dee69a401a04fc827ac2de5b7766e4", 736198ULL },
   { "openvino_detokenizer.xml", "ac37fdd9b044a04130747c6773b85f395e5910e0cfb132a2c937fb533d69b4da", 9699ULL },
   { "openvino_encoder_model.bin", "678fbeebf9856e533f2ea4515608d0dec6cb6abb0c3caf5169f03654365fb952", 345094560ULL },
   { "openvino_encoder_model.xml", "e42313feeb567ade3c8f0eb24ac18d118338559f1a53b475f7c8d31670dbcfb8", 1721425ULL },
   { "openvino_tokenizer.bin", "adfa3d9a2920d0f314121270a960ab331ec0f05838544bb8ecaaa422282a6fd4", 1898973ULL },
   { "openvino_tokenizer.xml", "dad2092c08510c043acabeae952cdbd381bc7b9c3272eaef524fa4fbea2f9dc4", 27011ULL },
   { "preprocessor_config.json", "654cf18d3e163b948ceaf9766da56ce0b52de265d58673cf61c9376f126bd499", 357ULL },
   { "special_tokens_map.json", "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74", 2186ULL },
   { "tokenizer.json", "5c1bf30c9e716e1477bedef846b01be0013daecb89e9e3ef7ab89b23c178df1b", 3930645ULL },
   { "tokenizer_config.json", "3c75940dfce3a294fca7041a5faff011677f1b68fa85e47511bb8cf6dccaded6", 282873ULL },
   { "vocab.json", "6788c80b082e9b0d1393147d3a3e62ba19285ac0c82ace8e5ef00f37ead58971", 835528ULL },
};

inline constexpr const char* const* kDependencies_whisper_large_v3_int4 = nullptr;

inline constexpr ModelFileInfo kFiles_distil_whisper_large_v3_fp16[] = {
   { "added_tokens.json", "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0", 34648ULL },
   { "config.json", "f2ad7fe724f34cca9c4a17ee3e8e9dd297461cedef357318179198ae00697b55", 1302ULL },
   { "generation_config.json", "1ace5fd5a162c94e76f170b372974db0dd44e60a3619ecbf09d1ab909fcf6490", 4242ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "4c11fc0c32a804cb351748873570567d4d4472a257879db8da78401e2a1f6875", 238873756ULL },
   { "openvino_decoder_model.xml", "cddf8f3cb5082c0deefac70de15206a51ee241de9fb87400c2acfceaacccaa07", 190521ULL },
   { "openvino_detokenizer.bin", "f2b3c47825a1089525ff65c0c8e49271e1dee69a401a04fc827ac2de5b7766e4", 736198ULL },
   { "openvino_detokenizer.xml", "0960740690766d773a6afa0f96e456381ceec1e667ca2951f440d084a2f91700", 9699ULL },
   { "openvino_encoder_model.bin", "76de5b22435747d7f0d7f74b003acd23ab5c10b7e1198d36373cdf0f6975b630", 1273938026ULL },
   { "openvino_encoder_model.xml", "d913c09a5c5c1f86e6b5dced2a44df0b69b9aacf8218451d9100ee4fa12dec3d", 1239276ULL },
   { "openvino_tokenizer.bin", "adfa3d9a2920d0f314121270a960ab331ec0f05838544bb8ecaaa422282a6fd4", 1898973ULL },
   { "openvino_tokenizer.xml", "c1b63607b10040e0ae94b248fec36098799d12ef921b98bb2116277c3840fed9", 27011ULL },
   { "preprocessor_config.json", "654cf18d3e163b948ceaf9766da56ce0b52de265d58673cf61c9376f126bd499", 357ULL },
   { "special_tokens_map.json", "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74", 2186ULL },
   { "tokenizer.json", "5c1bf30c9e716e1477bedef846b01be0013daecb89e9e3ef7ab89b23c178df1b", 3930645ULL },
   { "tokenizer_config.json", "3c75940dfce3a294fca7041a5faff011677f1b68fa85e47511bb8cf6dccaded6", 282873ULL },
   { "vocab.json", "6788c80b082e9b0d1393147d3a3e62ba19285ac0c82ace8e5ef00f37ead58971", 835528ULL },
};

inline constexpr const char* const* kDependencies_distil_whisper_large_v3_fp16 = nullptr;

inline constexpr ModelFileInfo kFiles_distil_whisper_large_v3_int8[] = {
   { "added_tokens.json", "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0", 34648ULL },
   { "config.json", "f2ad7fe724f34cca9c4a17ee3e8e9dd297461cedef357318179198ae00697b55", 1302ULL },
   { "generation_config.json", "1ace5fd5a162c94e76f170b372974db0dd44e60a3619ecbf09d1ab909fcf6490", 4242ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "15c407b7850fb126e8fe053dff744b9df1c312e8a3aca9818e43c7f4eab2c6d0", 119831990ULL },
   { "openvino_decoder_model.xml", "955b4a2c4c60909377d04a6d06b9b0a2acf49cd43c2aef81c8840c133ff3558e", 216009ULL },
   { "openvino_detokenizer.bin", "f2b3c47825a1089525ff65c0c8e49271e1dee69a401a04fc827ac2de5b7766e4", 736198ULL },
   { "openvino_detokenizer.xml", "c72f533f031414f0bd8ef83bd9b76e2a6a601edf3b8e8ea749c2a5b9c343ff73", 9699ULL },
   { "openvino_encoder_model.bin", "fffcbf47a4cfd5a1e3f57c0569f5ef706245b798ef626db5ffea7b84166ed865", 645332592ULL },
   { "openvino_encoder_model.xml", "961a70d37b7209ddf465a7646cec7f02606fbbb84850a1b7e65adb212477751d", 1518639ULL },
   { "openvino_tokenizer.bin", "adfa3d9a2920d0f314121270a960ab331ec0f05838544bb8ecaaa422282a6fd4", 1898973ULL },
   { "openvino_tokenizer.xml", "382c880c4b75b84a6469fc55f2b39c6bba4bed9457b9bf62ad6967a85526a915", 27011ULL },
   { "preprocessor_config.json", "654cf18d3e163b948ceaf9766da56ce0b52de265d58673cf61c9376f126bd499", 357ULL },
   { "special_tokens_map.json", "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74", 2186ULL },
   { "tokenizer.json", "5c1bf30c9e716e1477bedef846b01be0013daecb89e9e3ef7ab89b23c178df1b", 3930645ULL },
   { "tokenizer_config.json", "3c75940dfce3a294fca7041a5faff011677f1b68fa85e47511bb8cf6dccaded6", 282873ULL },
   { "vocab.json", "6788c80b082e9b0d1393147d3a3e62ba19285ac0c82ace8e5ef00f37ead58971", 835528ULL },
};

inline constexpr const char* const* kDependencies_distil_whisper_large_v3_int8 = nullptr;

inline constexpr ModelFileInfo kFiles_distil_whisper_large_v3_int4[] = {
   { "added_tokens.json", "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0", 34648ULL },
   { "config.json", "f2ad7fe724f34cca9c4a17ee3e8e9dd297461cedef357318179198ae00697b55", 1302ULL },
   { "generation_config.json", "1ace5fd5a162c94e76f170b372974db0dd44e60a3619ecbf09d1ab909fcf6490", 4242ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "f380be8ef88c032991fc80a3492ea8a2df2e465027b2453c209c099b121a4a53", 94541798ULL },
   { "openvino_decoder_model.xml", "9acb092cbb86645c3a390a57d6e51800907b0cf53dd5fbe6c83b5b77a8d0afd4", 237067ULL },
   { "openvino_detokenizer.bin", "f2b3c47825a1089525ff65c0c8e49271e1dee69a401a04fc827ac2de5b7766e4", 736198ULL },
   { "openvino_detokenizer.xml", "c72f533f031414f0bd8ef83bd9b76e2a6a601edf3b8e8ea749c2a5b9c343ff73", 9699ULL },
   { "openvino_encoder_model.bin", "678fbeebf9856e533f2ea4515608d0dec6cb6abb0c3caf5169f03654365fb952", 345094560ULL },
   { "openvino_encoder_model.xml", "7de1f16b6f475a5021423d7ff769546189fdc6bf75dc63a3f5c087dc40385dfc", 1721341ULL },
   { "openvino_tokenizer.bin", "adfa3d9a2920d0f314121270a960ab331ec0f05838544bb8ecaaa422282a6fd4", 1898973ULL },
   { "openvino_tokenizer.xml", "382c880c4b75b84a6469fc55f2b39c6bba4bed9457b9bf62ad6967a85526a915", 27011ULL },
   { "preprocessor_config.json", "654cf18d3e163b948ceaf9766da56ce0b52de265d58673cf61c9376f126bd499", 357ULL },
   { "special_tokens_map.json", "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74", 2186ULL },
   { "tokenizer.json", "5c1bf30c9e716e1477bedef846b01be0013daecb89e9e3ef7ab89b23c178df1b", 3930645ULL },
   { "tokenizer_config.json", "3c75940dfce3a294fca7041a5faff011677f1b68fa85e47511bb8cf6dccaded6", 282873ULL },
   { "vocab.json", "6788c80b082e9b0d1393147d3a3e62ba19285ac0c82ace8e5ef00f37ead58971", 835528ULL },
};

inline constexpr const char* const* kDependencies_distil_whisper_large_v3_int4 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_large_v3_turbo_fp16[] = {
   { "added_tokens.json", "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0", 34648ULL },
   { "config.json", "bfd92c097547ab12cb42abae8008be5a59a91fdc5ab39acce24489eb8a3e8a86", 1192ULL },
   { "generation_config.json", "4617fcca458af3b91a103143aaac919c1ab6680b552d7abd10811b7248bd77b4", 3767ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "58fbe879d2e028fc7792036ba660fde0e6c8dd85b71d596cd1a144672a807b73", 343818396ULL },
   { "openvino_decoder_model.xml", "926f88e8439e6850264ff14cb57f2737a0d87a8c5b7211870f985a78c69d68f4", 344388ULL },
   { "openvino_detokenizer.bin", "f2b3c47825a1089525ff65c0c8e49271e1dee69a401a04fc827ac2de5b7766e4", 736198ULL },
   { "openvino_detokenizer.xml", "cd4c080beed5a2db5b96aa7a50cf98b9e13695f99ffabeb57962893d9459fcc4", 9779ULL },
   { "openvino_encoder_model.bin", "d36aff56fa2215fd93ffe4e4eb17be8a97fbad16f97f6c96126d79161b5e9b83", 1273938026ULL },
   { "openvino_encoder_model.xml", "7bd76d5bd60795243ef4aa06b111ea58936d2df59885022b72415d6038d36e28", 1239274ULL },
   { "openvino_tokenizer.bin", "adfa3d9a2920d0f314121270a960ab331ec0f05838544bb8ecaaa422282a6fd4", 1898973ULL },
   { "openvino_tokenizer.xml", "79b781222585b8e45dc88791204c9b4bef84cfbce71c96ad505fc3870bd148b7", 27091ULL },
   { "preprocessor_config.json", "654cf18d3e163b948ceaf9766da56ce0b52de265d58673cf61c9376f126bd499", 357ULL },
   { "special_tokens_map.json", "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74", 2186ULL },
   { "tokenizer.json", "5c1bf30c9e716e1477bedef846b01be0013daecb89e9e3ef7ab89b23c178df1b", 3930645ULL },
   { "tokenizer_config.json", "3c75940dfce3a294fca7041a5faff011677f1b68fa85e47511bb8cf6dccaded6", 282873ULL },
   { "vocab.json", "6788c80b082e9b0d1393147d3a3e62ba19285ac0c82ace8e5ef00f37ead58971", 835528ULL },
};

inline constexpr const char* const* kDependencies_whisper_large_v3_turbo_fp16 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_large_v3_turbo_int8[] = {
   { "added_tokens.json", "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0", 34648ULL },
   { "config.json", "bfd92c097547ab12cb42abae8008be5a59a91fdc5ab39acce24489eb8a3e8a86", 1192ULL },
   { "generation_config.json", "4617fcca458af3b91a103143aaac919c1ab6680b552d7abd10811b7248bd77b4", 3767ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "c064991cbafc4381567d29972b7013dc24026de9c326d03eb1e6e4fc44aa959f", 172534710ULL },
   { "openvino_decoder_model.xml", "aeb09fafbf1c0cbf84baf30f46763436005822faf3b365359b8de0aa04f03047", 391642ULL },
   { "openvino_detokenizer.bin", "f2b3c47825a1089525ff65c0c8e49271e1dee69a401a04fc827ac2de5b7766e4", 736198ULL },
   { "openvino_detokenizer.xml", "6e106a14f14b0771b46b7948a99b1d819ff93b2455b7da8f47761ab9dba9dc56", 9779ULL },
   { "openvino_encoder_model.bin", "0590a8f35f96d57801c55990028d917821ac721026e34b7f3f59d7561fc908e6", 645332592ULL },
   { "openvino_encoder_model.xml", "60713d4ed3a8ac8ee020e11c4737ec276d14cabc6a082537bddf2c00ba6ce070", 1518660ULL },
   { "openvino_tokenizer.bin", "adfa3d9a2920d0f314121270a960ab331ec0f05838544bb8ecaaa422282a6fd4", 1898973ULL },
   { "openvino_tokenizer.xml", "cba304e7bad54773b9d2cbccfbc8501117ecf2e3c0f4f5331742a0a3c9feed93", 27091ULL },
   { "preprocessor_config.json", "654cf18d3e163b948ceaf9766da56ce0b52de265d58673cf61c9376f126bd499", 357ULL },
   { "special_tokens_map.json", "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74", 2186ULL },
   { "tokenizer.json", "5c1bf30c9e716e1477bedef846b01be0013daecb89e9e3ef7ab89b23c178df1b", 3930645ULL },
   { "tokenizer_config.json", "3c75940dfce3a294fca7041a5faff011677f1b68fa85e47511bb8cf6dccaded6", 282873ULL },
   { "vocab.json", "6788c80b082e9b0d1393147d3a3e62ba19285ac0c82ace8e5ef00f37ead58971", 835528ULL },
};

inline constexpr const char* const* kDependencies_whisper_large_v3_turbo_int8 = nullptr;

inline constexpr ModelFileInfo kFiles_whisper_large_v3_turbo_int4[] = {
   { "added_tokens.json", "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0", 34648ULL },
   { "config.json", "bfd92c097547ab12cb42abae8008be5a59a91fdc5ab39acce24489eb8a3e8a86", 1192ULL },
   { "generation_config.json", "4617fcca458af3b91a103143aaac919c1ab6680b552d7abd10811b7248bd77b4", 3767ULL },
   { "normalizer.json", "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd", 52666ULL },
   { "openvino_decoder_model.bin", "70a34bd26aa8b802535420239d61b8d6cc0876ef777046d395c788435bc5572e", 121954278ULL },
   { "openvino_decoder_model.xml", "8306639b2f235c3a2b9a8a7821c21fbdd096e0b1c107868b1dd29e57f2e2588c", 433734ULL },
   { "openvino_detokenizer.bin", "f2b3c47825a1089525ff65c0c8e49271e1dee69a401a04fc827ac2de5b7766e4", 736198ULL },
   { "openvino_detokenizer.xml", "6e106a14f14b0771b46b7948a99b1d819ff93b2455b7da8f47761ab9dba9dc56", 9779ULL },
   { "openvino_encoder_model.bin", "86edfce9765b229a6a47b0036ef2eb96ff898675d424c51fc2b165e088d3bbc4", 345094560ULL },
   { "openvino_encoder_model.xml", "ac2adae65fcc6f6ea958e3a180570e7eca3a453f448387cc703080db6ab88880", 1721370ULL },
   { "openvino_tokenizer.bin", "adfa3d9a2920d0f314121270a960ab331ec0f05838544bb8ecaaa422282a6fd4", 1898973ULL },
   { "openvino_tokenizer.xml", "cba304e7bad54773b9d2cbccfbc8501117ecf2e3c0f4f5331742a0a3c9feed93", 27091ULL },
   { "preprocessor_config.json", "654cf18d3e163b948ceaf9766da56ce0b52de265d58673cf61c9376f126bd499", 357ULL },
   { "special_tokens_map.json", "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74", 2186ULL },
   { "tokenizer.json", "5c1bf30c9e716e1477bedef846b01be0013daecb89e9e3ef7ab89b23c178df1b", 3930645ULL },
   { "tokenizer_config.json", "3c75940dfce3a294fca7041a5faff011677f1b68fa85e47511bb8cf6dccaded6", 282873ULL },
   { "vocab.json", "6788c80b082e9b0d1393147d3a3e62ba19285ac0c82ace8e5ef00f37ead58971", 835528ULL },
};

inline constexpr const char* const* kDependencies_whisper_large_v3_turbo_int4 = nullptr;

inline constexpr ModelInfo kModels[] = {
   {
      "Music Restoration",
      "music_restoration_apollo_mp3_jusperlee",
      "Apollo MP3 Restore (@JusperLee)",
      "music_restoration_apollo_mp3_jusperlee",
      "https://huggingface.co/Intel/apollo_jusperlee_openvino/resolve/720c90a7df79fd6add733ca9748a22b471a3bc09/",
      "720c90a7df79fd6add733ca9748a22b471a3bc09",
      "?download=true",
      "music_restoration/apollo_jusperlee",
      kDependencies_music_restoration_apollo_mp3_jusperlee,
      0,
      kFiles_music_restoration_apollo_mp3_jusperlee,
      sizeof(kFiles_music_restoration_apollo_mp3_jusperlee) / sizeof(kFiles_music_restoration_apollo_mp3_jusperlee[0])
   },
   {
      "Music Restoration",
      "music_restoration_apollo_universal_lew",
      "Apollo Universal Restore (@Lew)",
      "music_restoration_apollo_universal_lew",
      "",
      "",
      "?download=true",
      "music_restoration/apollo_universal",
      kDependencies_music_restoration_apollo_universal_lew,
      0,
      kFiles_music_restoration_apollo_universal_lew,
      sizeof(kFiles_music_restoration_apollo_universal_lew) / sizeof(kFiles_music_restoration_apollo_universal_lew[0])
   },
   {
      "Music Separation",
      "music_separation_demucs_v4",
      "Demucs v4",
      "music_separation_demucs_v4",
      "https://huggingface.co/Intel/demucs-openvino/resolve/3e9e7d2f15c1ff4877917a224f2f9668c9c41881/htdemucs_v4/",
      "3e9e7d2f15c1ff4877917a224f2f9668c9c41881",
      "?download=true",
      "stem_separation/htdemucs_v4",
      kDependencies_music_separation_demucs_v4,
      0,
      kFiles_music_separation_demucs_v4,
      sizeof(kFiles_music_separation_demucs_v4) / sizeof(kFiles_music_separation_demucs_v4[0])
   },
   {
      "Music Separation",
      "music_separation_demucs_v4_ft_drums",
      "Demucs v4 FT Drums",
      "music_separation_demucs_v4_ft_drums",
      "https://huggingface.co/Intel/demucs-openvino/resolve/3e9e7d2f15c1ff4877917a224f2f9668c9c41881/htdemucs_v4_ht_drums/",
      "3e9e7d2f15c1ff4877917a224f2f9668c9c41881",
      "?download=true",
      "stem_separation/htdemucs_v4_ht_drums",
      kDependencies_music_separation_demucs_v4_ft_drums,
      0,
      kFiles_music_separation_demucs_v4_ft_drums,
      sizeof(kFiles_music_separation_demucs_v4_ft_drums) / sizeof(kFiles_music_separation_demucs_v4_ft_drums[0])
   },
   {
      "Music Separation",
      "music_separation_demucs_v4_ft_bass",
      "Demucs v4 FT Bass",
      "music_separation_demucs_v4_ft_bass",
      "https://huggingface.co/Intel/demucs-openvino/resolve/3e9e7d2f15c1ff4877917a224f2f9668c9c41881/htdemucs_v4_ht_bass/",
      "3e9e7d2f15c1ff4877917a224f2f9668c9c41881",
      "?download=true",
      "stem_separation/htdemucs_v4_ht_bass",
      kDependencies_music_separation_demucs_v4_ft_bass,
      0,
      kFiles_music_separation_demucs_v4_ft_bass,
      sizeof(kFiles_music_separation_demucs_v4_ft_bass) / sizeof(kFiles_music_separation_demucs_v4_ft_bass[0])
   },
   {
      "Music Separation",
      "music_separation_demucs_v4_ft_other",
      "Demucs v4 FT Other Instruments",
      "music_separation_demucs_v4_ft_other",
      "https://huggingface.co/Intel/demucs-openvino/resolve/3e9e7d2f15c1ff4877917a224f2f9668c9c41881/htdemucs_v4_ht_other/",
      "3e9e7d2f15c1ff4877917a224f2f9668c9c41881",
      "?download=true",
      "stem_separation/htdemucs_v4_ht_other",
      kDependencies_music_separation_demucs_v4_ft_other,
      0,
      kFiles_music_separation_demucs_v4_ft_other,
      sizeof(kFiles_music_separation_demucs_v4_ft_other) / sizeof(kFiles_music_separation_demucs_v4_ft_other[0])
   },
   {
      "Music Separation",
      "music_separation_demucs_v4_ft_vocals",
      "Demucs v4 FT Vocals",
      "music_separation_demucs_v4_ft_vocals",
      "https://huggingface.co/Intel/demucs-openvino/resolve/3e9e7d2f15c1ff4877917a224f2f9668c9c41881/htdemucs_v4_ht_vocals/",
      "3e9e7d2f15c1ff4877917a224f2f9668c9c41881",
      "?download=true",
      "stem_separation/htdemucs_v4_ht_vocals",
      kDependencies_music_separation_demucs_v4_ft_vocals,
      0,
      kFiles_music_separation_demucs_v4_ft_vocals,
      sizeof(kFiles_music_separation_demucs_v4_ft_vocals) / sizeof(kFiles_music_separation_demucs_v4_ft_vocals[0])
   },
   {
      "Music Separation",
      "music_separation_demucs_v4_6s",
      "Demucs v4 6s",
      "music_separation_demucs_v4_6s",
      "https://huggingface.co/Intel/demucs-openvino/resolve/3e9e7d2f15c1ff4877917a224f2f9668c9c41881/htdemucs_v4_6s/",
      "3e9e7d2f15c1ff4877917a224f2f9668c9c41881",
      "?download=true",
      "stem_separation/htdemucs_v4_6s",
      kDependencies_music_separation_demucs_v4_6s,
      0,
      kFiles_music_separation_demucs_v4_6s,
      sizeof(kFiles_music_separation_demucs_v4_6s) / sizeof(kFiles_music_separation_demucs_v4_6s[0])
   },
   {
      "Music Separation",
      "music_separation_mel_vocals_kimberley_jenson",
      "MelBandRoformer Vocals (@KimberleyJensen)",
      "music_separation_mel_vocals_kimberley_jenson",
      "https://huggingface.co/Intel/vocals_mel_band_roformer_kimberleyJSN_openvino/resolve/ce2bae0e27f9b115f38b1ddad35439df2d28cbbd/",
      "ce2bae0e27f9b115f38b1ddad35439df2d28cbbd",
      "?download=true",
      "stem_separation/melband_roformer_kimberley_jenson",
      kDependencies_music_separation_mel_vocals_kimberley_jenson,
      0,
      kFiles_music_separation_mel_vocals_kimberley_jenson,
      sizeof(kFiles_music_separation_mel_vocals_kimberley_jenson) / sizeof(kFiles_music_separation_mel_vocals_kimberley_jenson[0])
   },
   {
      "Music Separation",
      "music_separation_mel_crowd_aufr33_viperx",
      "MelBandRoformer Crowd (@aufr33, @viperx)",
      "music_separation_mel_crowd_aufr33_viperx",
      "https://huggingface.co/Intel/crowd_mel_band_roformer_aufr33_viperx_openvino/resolve/b35f0dc8e9ee507582bc93a6e2b52e0dba9eca93/",
      "b35f0dc8e9ee507582bc93a6e2b52e0dba9eca93",
      "?download=true",
      "stem_separation/melband_roformer_crowd",
      kDependencies_music_separation_mel_crowd_aufr33_viperx,
      0,
      kFiles_music_separation_mel_crowd_aufr33_viperx,
      sizeof(kFiles_music_separation_mel_crowd_aufr33_viperx) / sizeof(kFiles_music_separation_mel_crowd_aufr33_viperx[0])
   },
   {
      "Music Separation",
      "music_separation_msdx23c_drum_sep_jarredou",
      "MDX23C Drum Separation (@jarredou)",
      "music_separation_msdx23c_drum_sep_jarredou",
      "https://huggingface.co/Intel/drumsep_mdx23c_jarredou_openvino/resolve/2944425500506842ccc4ca130b22be8cfe95b20d/",
      "2944425500506842ccc4ca130b22be8cfe95b20d",
      "?download=true",
      "stem_separation/drumsep_jarredou_mdx23c",
      kDependencies_music_separation_msdx23c_drum_sep_jarredou,
      0,
      kFiles_music_separation_msdx23c_drum_sep_jarredou,
      sizeof(kFiles_music_separation_msdx23c_drum_sep_jarredou) / sizeof(kFiles_music_separation_msdx23c_drum_sep_jarredou[0])
   },
   {
      "Noise Suppression",
      "noise_suppression_deepfilternet2",
      "DeepFilterNet2",
      "noise_suppression_deepfilternet2",
      "https://huggingface.co/Intel/deepfilternet-openvino/resolve/0615a1b18be8585156130a98fdbca75e9719eda3/deepfilternet2/",
      "0615a1b18be8585156130a98fdbca75e9719eda3",
      "?download=true",
      "noise_suppression/deepfilternet2",
      kDependencies_noise_suppression_deepfilternet2,
      0,
      kFiles_noise_suppression_deepfilternet2,
      sizeof(kFiles_noise_suppression_deepfilternet2) / sizeof(kFiles_noise_suppression_deepfilternet2[0])
   },
   {
      "Noise Suppression",
      "noise_suppression_deepfilternet3",
      "DeepFilterNet3",
      "noise_suppression_deepfilternet3",
      "https://huggingface.co/Intel/deepfilternet-openvino/resolve/0615a1b18be8585156130a98fdbca75e9719eda3/deepfilternet3/",
      "0615a1b18be8585156130a98fdbca75e9719eda3",
      "?download=true",
      "noise_suppression/deepfilternet3",
      kDependencies_noise_suppression_deepfilternet3,
      0,
      kFiles_noise_suppression_deepfilternet3,
      sizeof(kFiles_noise_suppression_deepfilternet3) / sizeof(kFiles_noise_suppression_deepfilternet3[0])
   },
   {
      "Noise Suppression",
      "noise_suppression_denseunet",
      "DenseUNet",
      "noise_suppression/noise_suppression_denseunet",
      "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16/",
      "2023.0",
      "",
      "noise_suppression/denseunet",
      kDependencies_noise_suppression_denseunet,
      0,
      kFiles_noise_suppression_denseunet,
      sizeof(kFiles_noise_suppression_denseunet) / sizeof(kFiles_noise_suppression_denseunet[0])
   },
   {
      "Reverb Removal",
      "reverb_removal_mel_band_roformer_mono_anvuew",
      "MelBandRoformer Dereverb Mono (@anvuew)",
      "reverb_removal_mel_band_dereverb_mono_anvuew",
      "https://huggingface.co/Intel/dereverb_mel_band_roformer_anvuew_openvino/resolve/16aeb6904702657415c04bdc906dc9c3ed6524a1/mono/",
      "16aeb6904702657415c04bdc906dc9c3ed6524a1",
      "?download=true",
      "reverb_removal/mel_band_roformer_mono_anvuew",
      kDependencies_reverb_removal_mel_band_roformer_mono_anvuew,
      0,
      kFiles_reverb_removal_mel_band_roformer_mono_anvuew,
      sizeof(kFiles_reverb_removal_mel_band_roformer_mono_anvuew) / sizeof(kFiles_reverb_removal_mel_band_roformer_mono_anvuew[0])
   },
   {
      "Super Resolution",
      "super_resolution_common",
      "Super Resolution Common",
      "super_resolution_common",
      "https://huggingface.co/Intel/versatile_audio_super_resolution_openvino/resolve/b98a5a9e21ede61cd556cd04d425bf5bfd675328/",
      "b98a5a9e21ede61cd556cd04d425bf5bfd675328",
      "?download=true",
      "audiosr",
      kDependencies_super_resolution_common,
      0,
      kFiles_super_resolution_common,
      sizeof(kFiles_super_resolution_common) / sizeof(kFiles_super_resolution_common[0])
   },
   {
      "Super Resolution",
      "super_resolution_basic_general_fp16",
      "Basic (General) (FP16)",
      "super_resolution_basic_general",
      "https://huggingface.co/Intel/versatile_audio_super_resolution_openvino/resolve/b98a5a9e21ede61cd556cd04d425bf5bfd675328/",
      "b98a5a9e21ede61cd556cd04d425bf5bfd675328",
      "?download=true",
      "audiosr",
      kDependencies_super_resolution_basic_general_fp16,
      1,
      kFiles_super_resolution_basic_general_fp16,
      sizeof(kFiles_super_resolution_basic_general_fp16) / sizeof(kFiles_super_resolution_basic_general_fp16[0])
   },
   {
      "Super Resolution",
      "super_resolution_speech_fp16",
      "Speech (FP16)",
      "super_resolution_speech",
      "https://huggingface.co/Intel/versatile_audio_super_resolution_openvino/resolve/b98a5a9e21ede61cd556cd04d425bf5bfd675328/",
      "b98a5a9e21ede61cd556cd04d425bf5bfd675328",
      "?download=true",
      "audiosr",
      kDependencies_super_resolution_speech_fp16,
      1,
      kFiles_super_resolution_speech_fp16,
      sizeof(kFiles_super_resolution_speech_fp16) / sizeof(kFiles_super_resolution_speech_fp16[0])
   },
   {
      "Text-to-Speech",
      "text_to_speech_kokoro_82m",
      "Kokoro-82M",
      "text_to_speech_kokoro_82m",
      "",
      "",
      "?download=true",
      "text_to_speech/ov_Kokoro-82M",
      kDependencies_text_to_speech_kokoro_82m,
      0,
      kFiles_text_to_speech_kokoro_82m,
      sizeof(kFiles_text_to_speech_kokoro_82m) / sizeof(kFiles_text_to_speech_kokoro_82m[0])
   },
   {
      "Whisper Transcription",
      "whisper_base_fp16",
      "Whisper Base (FP16)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-base-fp16-ov/resolve/84fbe975a79a8c996fd32c036558f29e2db6670f/",
      "84fbe975a79a8c996fd32c036558f29e2db6670f",
      "?download=true",
      "whisper/whisper-base-fp16-ov",
      kDependencies_whisper_base_fp16,
      0,
      kFiles_whisper_base_fp16,
      sizeof(kFiles_whisper_base_fp16) / sizeof(kFiles_whisper_base_fp16[0])
   },
   {
      "Whisper Transcription",
      "whisper_base_int8",
      "Whisper Base (INT8)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-base-int8-ov/resolve/0606293f0511136ada21755a265492f623a934b8/",
      "0606293f0511136ada21755a265492f623a934b8",
      "?download=true",
      "whisper/whisper-base-int8-ov",
      kDependencies_whisper_base_int8,
      0,
      kFiles_whisper_base_int8,
      sizeof(kFiles_whisper_base_int8) / sizeof(kFiles_whisper_base_int8[0])
   },
   {
      "Whisper Transcription",
      "whisper_base_int4",
      "Whisper Base (INT4)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-base-int4-ov/resolve/21b22adb8e49b79dab004804a1b40655a4767c37/",
      "21b22adb8e49b79dab004804a1b40655a4767c37",
      "?download=true",
      "whisper/whisper-base-int4-ov",
      kDependencies_whisper_base_int4,
      0,
      kFiles_whisper_base_int4,
      sizeof(kFiles_whisper_base_int4) / sizeof(kFiles_whisper_base_int4[0])
   },
   {
      "Whisper Transcription",
      "whisper_medium_fp16",
      "Whisper Medium (FP16)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-medium-fp16-ov/resolve/4508616c9c0774807e7d315c26cec49dcfe1f0a8/",
      "4508616c9c0774807e7d315c26cec49dcfe1f0a8",
      "?download=true",
      "whisper/whisper-medium-fp16-ov",
      kDependencies_whisper_medium_fp16,
      0,
      kFiles_whisper_medium_fp16,
      sizeof(kFiles_whisper_medium_fp16) / sizeof(kFiles_whisper_medium_fp16[0])
   },
   {
      "Whisper Transcription",
      "whisper_medium_int8",
      "Whisper Medium (INT8)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-medium-int8-ov/resolve/8d43cce846729381f56bd45a1c70925cee2222ff/",
      "8d43cce846729381f56bd45a1c70925cee2222ff",
      "?download=true",
      "whisper/whisper-medium-int8-ov",
      kDependencies_whisper_medium_int8,
      0,
      kFiles_whisper_medium_int8,
      sizeof(kFiles_whisper_medium_int8) / sizeof(kFiles_whisper_medium_int8[0])
   },
   {
      "Whisper Transcription",
      "whisper_medium_int4",
      "Whisper Medium (INT4)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-medium-int4-ov/resolve/14bba652dc6604717bf1cbdf358645d414548522/",
      "14bba652dc6604717bf1cbdf358645d414548522",
      "?download=true",
      "whisper/whisper-medium-int4-ov",
      kDependencies_whisper_medium_int4,
      0,
      kFiles_whisper_medium_int4,
      sizeof(kFiles_whisper_medium_int4) / sizeof(kFiles_whisper_medium_int4[0])
   },
   {
      "Whisper Transcription",
      "whisper_large_v2_fp16",
      "Whisper Large V2 (FP16)",
      "whisper_transcription_info",
      "",
      "",
      "?download=true",
      "whisper/whisper-large-v2-fp16-ov",
      kDependencies_whisper_large_v2_fp16,
      0,
      kFiles_whisper_large_v2_fp16,
      sizeof(kFiles_whisper_large_v2_fp16) / sizeof(kFiles_whisper_large_v2_fp16[0])
   },
   {
      "Whisper Transcription",
      "whisper_large_v2_int8",
      "Whisper Large V2 (INT8)",
      "whisper_transcription_info",
      "",
      "",
      "?download=true",
      "whisper/whisper-large-v2-int8-ov",
      kDependencies_whisper_large_v2_int8,
      0,
      kFiles_whisper_large_v2_int8,
      sizeof(kFiles_whisper_large_v2_int8) / sizeof(kFiles_whisper_large_v2_int8[0])
   },
   {
      "Whisper Transcription",
      "whisper_large_v2_int4",
      "Whisper Large V2 (INT4)",
      "whisper_transcription_info",
      "",
      "",
      "?download=true",
      "whisper/whisper-large-v2-int4-ov",
      kDependencies_whisper_large_v2_int4,
      0,
      kFiles_whisper_large_v2_int4,
      sizeof(kFiles_whisper_large_v2_int4) / sizeof(kFiles_whisper_large_v2_int4[0])
   },
   {
      "Whisper Transcription",
      "whisper_large_v3_fp16",
      "Whisper Large V3 (FP16)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-large-v3-fp16-ov/resolve/220761e60602a5ca694c409d5f424563b75d6820/",
      "220761e60602a5ca694c409d5f424563b75d6820",
      "?download=true",
      "whisper/whisper-large-v3-fp16-ov",
      kDependencies_whisper_large_v3_fp16,
      0,
      kFiles_whisper_large_v3_fp16,
      sizeof(kFiles_whisper_large_v3_fp16) / sizeof(kFiles_whisper_large_v3_fp16[0])
   },
   {
      "Whisper Transcription",
      "whisper_large_v3_int8",
      "Whisper Large V3 (INT8)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-large-v3-int8-ov/resolve/a888a75cc8b494a8a45400fd85f6bfa379ba3955/",
      "a888a75cc8b494a8a45400fd85f6bfa379ba3955",
      "?download=true",
      "whisper/whisper-large-v3-int8-ov",
      kDependencies_whisper_large_v3_int8,
      0,
      kFiles_whisper_large_v3_int8,
      sizeof(kFiles_whisper_large_v3_int8) / sizeof(kFiles_whisper_large_v3_int8[0])
   },
   {
      "Whisper Transcription",
      "whisper_large_v3_int4",
      "Whisper Large V3 (INT4)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-large-v3-int4-ov/resolve/95f08bc1b2b53dafaecae3d806b056adecc0be33/",
      "95f08bc1b2b53dafaecae3d806b056adecc0be33",
      "?download=true",
      "whisper/whisper-large-v3-int4-ov",
      kDependencies_whisper_large_v3_int4,
      0,
      kFiles_whisper_large_v3_int4,
      sizeof(kFiles_whisper_large_v3_int4) / sizeof(kFiles_whisper_large_v3_int4[0])
   },
   {
      "Whisper Transcription",
      "distil_whisper_large_v3_fp16",
      "Distil-Whisper Large V3 (FP16)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/distil-whisper-large-v3-fp16-ov/resolve/147fc406c025905fa774599450c3ca98b72e5671/",
      "147fc406c025905fa774599450c3ca98b72e5671",
      "?download=true",
      "whisper/distil-whisper-large-v3-fp16-ov",
      kDependencies_distil_whisper_large_v3_fp16,
      0,
      kFiles_distil_whisper_large_v3_fp16,
      sizeof(kFiles_distil_whisper_large_v3_fp16) / sizeof(kFiles_distil_whisper_large_v3_fp16[0])
   },
   {
      "Whisper Transcription",
      "distil_whisper_large_v3_int8",
      "Distil-Whisper Large V3 (INT8)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/distil-whisper-large-v3-int8-ov/resolve/ab5db836c48303e296237013d7385924f2828e9d/",
      "ab5db836c48303e296237013d7385924f2828e9d",
      "?download=true",
      "whisper/distil-whisper-large-v3-int8-ov",
      kDependencies_distil_whisper_large_v3_int8,
      0,
      kFiles_distil_whisper_large_v3_int8,
      sizeof(kFiles_distil_whisper_large_v3_int8) / sizeof(kFiles_distil_whisper_large_v3_int8[0])
   },
   {
      "Whisper Transcription",
      "distil_whisper_large_v3_int4",
      "Distil-Whisper Large V3 (INT4)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/distil-whisper-large-v3-int4-ov/resolve/954b8ce3ca0e1d668d6ec41ea2b03e8420d95158/",
      "954b8ce3ca0e1d668d6ec41ea2b03e8420d95158",
      "?download=true",
      "whisper/distil-whisper-large-v3-int4-ov",
      kDependencies_distil_whisper_large_v3_int4,
      0,
      kFiles_distil_whisper_large_v3_int4,
      sizeof(kFiles_distil_whisper_large_v3_int4) / sizeof(kFiles_distil_whisper_large_v3_int4[0])
   },
   {
      "Whisper Transcription",
      "whisper_large_v3_turbo_fp16",
      "Whisper Large V3 Turbo (FP16)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-large-v3-turbo-fp16-ov/resolve/131d663658f94202779b0bb98ee7a5f71d5bde1a/",
      "131d663658f94202779b0bb98ee7a5f71d5bde1a",
      "?download=true",
      "whisper/whisper-large-v3-turbo-fp16-ov",
      kDependencies_whisper_large_v3_turbo_fp16,
      0,
      kFiles_whisper_large_v3_turbo_fp16,
      sizeof(kFiles_whisper_large_v3_turbo_fp16) / sizeof(kFiles_whisper_large_v3_turbo_fp16[0])
   },
   {
      "Whisper Transcription",
      "whisper_large_v3_turbo_int8",
      "Whisper Large V3 Turbo (INT8)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int8-ov/resolve/4929ae83ea2d1df59f4b5898a9aab8aa1c29e711/",
      "4929ae83ea2d1df59f4b5898a9aab8aa1c29e711",
      "?download=true",
      "whisper/whisper-large-v3-turbo-int8-ov",
      kDependencies_whisper_large_v3_turbo_int8,
      0,
      kFiles_whisper_large_v3_turbo_int8,
      sizeof(kFiles_whisper_large_v3_turbo_int8) / sizeof(kFiles_whisper_large_v3_turbo_int8[0])
   },
   {
      "Whisper Transcription",
      "whisper_large_v3_turbo_int4",
      "Whisper Large V3 Turbo (INT4)",
      "whisper_transcription_info",
      "https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int4-ov/resolve/ae50b4d9a9dbaf16f2df59c23f3984e42f864dfc/",
      "ae50b4d9a9dbaf16f2df59c23f3984e42f864dfc",
      "?download=true",
      "whisper/whisper-large-v3-turbo-int4-ov",
      kDependencies_whisper_large_v3_turbo_int4,
      0,
      kFiles_whisper_large_v3_turbo_int4,
      sizeof(kFiles_whisper_large_v3_turbo_int4) / sizeof(kFiles_whisper_large_v3_turbo_int4[0])
   },
};

inline constexpr std::size_t kModelCount = sizeof(kModels) / sizeof(kModels[0]);

} // namespace model_download_manifest
