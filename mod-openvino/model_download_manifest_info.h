// Auto-generated header containing model download metadata
#pragma once

#include <cstddef>

namespace model_download_manifest {

struct ModelFileInfo
{
   const char* name;
   const char* expected_sha256;
};

struct ModelInfo
{
   const char* effect;
   const char* model_id;
   const char* model_name;
   const char* info_key;
   const char* base_url;
   const char* post_url;
   const char* relative_path;
   const char* const* dependencies;
   std::size_t dependency_count;
   const ModelFileInfo* files;
   std::size_t file_count;
};

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4[] = {
   { "htdemucs_fwd.bin", "74c4e1ebd68b648ea5a2aabefb65d5c3ba805658037fc97a37b84ce9c72f4eb1" },
   { "htdemucs_fwd.xml", "1557f96f11cf1667beca60679926bf43b393d3ff71eb4b9a625c6a07bc23f4a4" },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4 = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4_ft_drums[] = {
   { "htdemucs_fwd.bin", "340e5f6d8dd4d5d0d987f545b922a8252cc5ea6e53649f91eadd8b36e77fcbac" },
   { "htdemucs_fwd.xml", "1557f96f11cf1667beca60679926bf43b393d3ff71eb4b9a625c6a07bc23f4a4" },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4_ft_drums = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4_ft_bass[] = {
   { "htdemucs_fwd.bin", "a7ec34d838b0f8f36b11a90d6f4e588141a383faadc77e9648e19fdd9eefc95e" },
   { "htdemucs_fwd.xml", "1557f96f11cf1667beca60679926bf43b393d3ff71eb4b9a625c6a07bc23f4a4" },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4_ft_bass = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4_ft_other[] = {
   { "htdemucs_fwd.bin", "c5f4bb99e91fea0bda75f64086712884457a5c9f861d1c2a7a02e09e92c2846e" },
   { "htdemucs_fwd.xml", "1557f96f11cf1667beca60679926bf43b393d3ff71eb4b9a625c6a07bc23f4a4" },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4_ft_other = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4_ft_vocals[] = {
   { "htdemucs_fwd.bin", "8576b4bd27095bc7f2dae39c529124b04eebc3e04009a45b7cc170f5b563f749" },
   { "htdemucs_fwd.xml", "1557f96f11cf1667beca60679926bf43b393d3ff71eb4b9a625c6a07bc23f4a4" },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4_ft_vocals = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_demucs_v4_6s[] = {
   { "htdemucs_fwd.bin", "5d66281099b5fdaba8844e306f3d6710b3e0b005fa9aad4514ce7604e24fbec9" },
   { "htdemucs_fwd.xml", "06c68a8a21a6cefb905575cbe689cfa3241416a4c263271804ae5690f8ba5cbf" },
};

inline constexpr const char* const* kDependencies_music_separation_demucs_v4_6s = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_mel_vocals_kimberley_jenson[] = {
   { "mel_band_pre.xml", "863afb676b3401a0030b3e84aa34db436d553491bf490b2e3f2ea04fea2a03f0" },
   { "mel_band_pre.bin", "ded35e388bb5662a161003d4036534a03cd7efa45e3c0c1383f1f1acb73d7f9a" },
   { "mel_band_post.xml", "b87793f6eedb7cb66d0e6c9efbb23b130ad494f360689ef32a5c19afd906320e" },
   { "mel_band_post.bin", "b7ebaa78cc9468995474b9a05c9887079fb806f4c861c834b1e9161effde447c" },
   { "mel_band_fwd.xml", "e9f681f7184f542338e3c963ba837fd2e622e08bf27e6592aeae92fca757efa4" },
   { "mel_band_fwd.bin", "1a23e353c7b41267ef683beb9be2d1c5f25a7cc2725b43f6beee68cb7cfc1a0b" },
};

inline constexpr const char* const* kDependencies_music_separation_mel_vocals_kimberley_jenson = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_mel_crowd_aufr33_viperx[] = {
   { "mel_band_pre.xml", "863afb676b3401a0030b3e84aa34db436d553491bf490b2e3f2ea04fea2a03f0" },
   { "mel_band_pre.bin", "ded35e388bb5662a161003d4036534a03cd7efa45e3c0c1383f1f1acb73d7f9a" },
   { "mel_band_post.xml", "b87793f6eedb7cb66d0e6c9efbb23b130ad494f360689ef32a5c19afd906320e" },
   { "mel_band_post.bin", "b7ebaa78cc9468995474b9a05c9887079fb806f4c861c834b1e9161effde447c" },
   { "mel_band_fwd.xml", "e9f681f7184f542338e3c963ba837fd2e622e08bf27e6592aeae92fca757efa4" },
   { "mel_band_fwd.bin", "bf585466ba9b3666d16ad18e0ca40cba59c6bf67a7017a95014d81f8aec0317a" },
};

inline constexpr const char* const* kDependencies_music_separation_mel_crowd_aufr33_viperx = nullptr;

inline constexpr ModelFileInfo kFiles_music_separation_msdx23c_drum_sep_jarredou[] = {
   { "mdx23c_fwd.xml", "62c7107ad49bac24bb4e0839971387d6053446c6aa234cf99bce7af7f6ec532c" },
   { "mdx23c_fwd.bin", "682f8616845a9f31bb21df38198df2d36ef79e96182d0ac7671d332afc80270e" },
};

inline constexpr const char* const* kDependencies_music_separation_msdx23c_drum_sep_jarredou = nullptr;

inline constexpr ModelInfo kModels[] = {
   {
      "Music Separation",
      "music_separation_demucs_v4",
      "Demucs v4",
      "music_separation_demucs_v4",
      "https://huggingface.co/Intel/demucs-openvino/resolve/3e9e7d2f15c1ff4877917a224f2f9668c9c41881/htdemucs_v4/",
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
      "?download=true",
      "stem_separation/drumsep_jarredou_mdx23c",
      kDependencies_music_separation_msdx23c_drum_sep_jarredou,
      0,
      kFiles_music_separation_msdx23c_drum_sep_jarredou,
      sizeof(kFiles_music_separation_msdx23c_drum_sep_jarredou) / sizeof(kFiles_music_separation_msdx23c_drum_sep_jarredou[0])
   },
};

inline constexpr std::size_t kModelCount = sizeof(kModels) / sizeof(kModels[0]);

} // namespace model_download_manifest
