// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "deepfilter.h"
#include "musicgen_utils.h"


namespace ov_deepfilternet
{
   DeepFilter::DeepFilter(std::string model_folder, std::string device, ModelSelection model_selection, std::optional<std::string> openvino_cache_dir,
      int64_t sr, int64_t fft_size, int64_t hop_size, int64_t nb_bands, int64_t min_nb_freqs, int64_t nb_df, double alpha)
   {
      _fft_size = fft_size;
      _frame_size = hop_size;
      _window_size = fft_size;
      _window_size_h = fft_size / 2;
      _freq_size = fft_size / 2 + 1;
      _wnorm = 1.f / ((_window_size * _window_size) / (2 * _frame_size));

      // Initialize the vorbis window: sin(pi/2*sin^2(pi*n/N))
      _window = torch::sin(0.5 * M_PI * (torch::arange(_fft_size) + 0.5) / _window_size_h);
      _window = torch::sin(0.5 * M_PI * (_window * _window));

      _sr = sr;
      _min_nb_freqs = min_nb_freqs;
      _nb_df = nb_df;

      // Initializing erb features
      _erb_indices = torch::tensor({
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8,
      10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67
         }, torch::dtype(torch::kInt64));
      _n_erb_features = nb_bands;

      // Create the convolutional layer
      torch::nn::Conv1dOptions conv_options(_freq_size, _n_erb_features, 1);
      conv_options.bias(false); // Set bias to false
      _erb_conv = std::make_shared<torch::nn::Conv1d>(conv_options);

      // Set requires_grad to false for all parameters
      for (auto& param : (*_erb_conv)->named_parameters()) {
         param.value().set_requires_grad(false);
      }

      // Initialize out_weight tensor
      auto out_weight = torch::zeros({ _n_erb_features, _freq_size, 1 });

      // Update out_weight based on erb_indices
      int64_t start_index = 0;
      for (int64_t i = 0; i < _erb_indices.size(0); ++i) {
         int64_t num = _erb_indices[i].item<int64_t>();
         out_weight.index({ i, torch::indexing::Slice(start_index, start_index + num), 0 }) = 1.0 / static_cast<double>(num);
         start_index += num;
      }

      // Copy out_weight to erb_conv's weight
      (*_erb_conv)->weight.copy_(out_weight);

      _mean_norm_init = { -60., -90. };
      _unit_norm_init = { 0.001, 0.0001 };

      _alpha = alpha;

      // Init buffers
      _reset_reg();

      _dfnet = std::make_shared< DFNetModel >(model_folder, device, model_selection, openvino_cache_dir, _erb_indices, 2, _nb_df);
   }

   void DeepFilter::_reset_reg()
   {
      _reg_analysis_mem = torch::zeros(_fft_size - _frame_size);
      _reg_synthesis_mem = torch::zeros(_fft_size - _frame_size);
      _reg_band_unit_norm_state = torch::linspace(_unit_norm_init[0], _unit_norm_init[1], _nb_df);
      _reg_erb_norm_state = torch::linspace(_mean_norm_init[0], _mean_norm_init[1], _n_erb_features);
   }

   std::shared_ptr<std::vector<float>> DeepFilter::filter(torch::Tensor noisy_audio, std::optional<float> atten_lim_db, float normalize_atten_lim, float df3_post_filter, ProgressCallbackFunc callback, void* callback_user)
   {

      //TODO: Make this less confusing. The hardcoded 2 here is lookahead.
      // Calculate the 'chunk size' which is the amount of samples that we *always* will pass into forward.
      const int64_t chunk_size = (_dfnet->num_static_hops() - 2) * _frame_size;

      std::cout << "chunk_size = " << chunk_size << std::endl;
      bool pad = true;

      //take the mean first, for the entire audio segment.
      //noisy_audio = noisy_audio.mean(0);
      noisy_audio = noisy_audio.squeeze(0);

      //this might be overkill, but it seems to provide good results... and inference on a 30-second
      // snippet is super fast, so why not.
      // 
      //TODO! Even though we overlap 10 seconds, only crossfade half of that (i.e. throw away entirely half of new segment).
      // The start of each new segment has a zero-ed out initial state, and could potentially have weird 'starting' artifacts.
      double crossfade_overlap_seconds = 10;

      //note, 48000 here is sample rate
      int64_t overlap_samples = (int64_t)((double)48000 * (double)crossfade_overlap_seconds);

      // each segment consists of an offset, and size
      std::vector< std::pair<int64_t, int64_t> > segments;

      int64_t total_nsamples = noisy_audio.size(0);

      {
         int64_t current_sample = 0;
         while (current_sample < total_nsamples)
         {
            if (current_sample != 0)
            {
               current_sample -= overlap_samples;
            }

            std::pair<size_t, size_t> segment = { current_sample,
               std::min(total_nsamples - current_sample, chunk_size) };

            segments.push_back(segment);

            current_sample += chunk_size;
         }
      }

      std::shared_ptr< std::vector<float> > output_wav = std::make_shared < std::vector<float> >(total_nsamples);

      for (size_t segmenti = 0; segmenti < segments.size(); segmenti++)
      {

         using namespace torch::indexing;

         auto src_offset = segments[segmenti].first;
         auto src_size = segments[segmenti].second;

         torch::Tensor chunk_tensor;
         if (src_size < chunk_size)
         {
            chunk_tensor = torch::zeros({ chunk_size });
            chunk_tensor.index_put_({ Slice(0, src_size) }, noisy_audio.index({ Slice(src_offset, src_offset + src_size) }));
         }
         else
         {
            chunk_tensor = noisy_audio.index({ Slice(src_offset, src_offset + src_size) });
         }

         auto wav = forward(chunk_tensor, pad, atten_lim_db, normalize_atten_lim, df3_post_filter);

         if (segmenti == 0)
         {
            std::memcpy(output_wav->data(), wav->data(), src_size * sizeof(float));
         }
         else
         {
            //cross-fade
            float fade_step = 1.f / (float)(overlap_samples);

            float* pLast = output_wav->data() + src_offset;
            float* pNew = wav->data();

            int64_t valid_overlap_samples = std::min(overlap_samples, src_size);
            torch::Tensor crossfade_left = torch::from_blob(pLast, { valid_overlap_samples });
            torch::Tensor crossfade_right = torch::from_blob(pNew, { valid_overlap_samples });

            auto multiplier = torch::linspace(0.f, 1.f, std::min(valid_overlap_samples, src_size));

            auto faded = crossfade_right * multiplier + crossfade_left * (1.f - multiplier);

            std::memcpy(pLast, faded.data_ptr(), valid_overlap_samples * sizeof(float));

            size_t samples_left = src_size - valid_overlap_samples;
            if (samples_left)
            {
               std::memcpy(pLast + valid_overlap_samples, pNew + valid_overlap_samples, samples_left * sizeof(float));
            }
         }

         if (callback)
         {
            float perc_complete = (float)(segmenti + 1) / segments.size();
            if (!callback(perc_complete, callback_user))
            {
               //callback returned false, so return empty wav
               return {};
            }
         }
      }

      return output_wav;
   }

   static inline torch::Tensor as_complex(torch::Tensor x)
   {
      if (torch::is_complex(x))
      {
         return x;
      }

      if (x.size(-1) != 2)
      {
         throw std::runtime_error("Last dimension need to be of length 2 (re + im)");
      }

      if (x.stride(-1) != 1)
      {
         x - x.contiguous();
      }

      return torch::view_as_complex(x);
   }

   std::shared_ptr<std::vector<float>> DeepFilter::forward(torch::Tensor noisy_audio, bool pad, std::optional<float> atten_lim_db, float normalize_atten_lim, float df3_post_filter)
   {
      _reset_reg();

      auto audio = noisy_audio;
      auto orig_len = audio.sizes().back();

      if (pad)
      {
         // Pad audio to compensate for the delay due to the real-time STFT implementation
         audio = torch::nn::functional::pad(audio, torch::nn::functional::PadFuncOptions({ 0, _fft_size }));
      }

      auto df_ret = _df_features(audio);
      auto spec = std::get<0>(df_ret);
      auto erb_feat = std::get<1>(df_ret);
      auto spec_feat = std::get<2>(df_ret);

      //_infer_request
      spec = spec.contiguous();
      erb_feat = erb_feat.contiguous();
      spec_feat = spec_feat.contiguous();

      auto enhanced = _dfnet->forward(spec, erb_feat, spec_feat, df3_post_filter);

      if (atten_lim_db && (std::abs(*atten_lim_db) > 0))
      {
         float lim = std::pow(10, (-std::abs(*atten_lim_db) / normalize_atten_lim));
         enhanced = torch::lerp(enhanced, spec, lim);
      }


      {
         std::vector< int64_t > view_shape;
         for (size_t i = 2; i < enhanced.sizes().size(); i++)
            view_shape.push_back(enhanced.size(i));
         enhanced = torch::view_as_complex(enhanced.view(view_shape));
      }

      audio = _synthesis_time(enhanced).unsqueeze(0);

      if (pad)
      {
         using namespace torch::indexing;
         auto d = _fft_size - _frame_size;
         audio = audio.index({ Slice(None), Slice(d, orig_len + d) });
      }

      std::shared_ptr<std::vector<float>> wav = std::make_shared< std::vector<float> >(audio.size(1));
      std::memcpy(wav->data(), audio.data_ptr<float>(), audio.size(1) * sizeof(float));
      return wav;
   }


   std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> DeepFilter::_df_features(torch::Tensor audio)
   {
      auto spec = _analysis_time(audio);
      auto erb_feat = _erb_norm_time(_erb(spec), _alpha);

      using namespace torch::indexing;
      auto spec_feat = torch::view_as_real(_unit_norm_time(spec.index({ "...", Slice(None, _nb_df) }), _alpha));
      spec = torch::view_as_real(spec);

      auto spec_ret = spec.unsqueeze(0).unsqueeze(0);
      auto erb_feat_ret = erb_feat.unsqueeze(0).unsqueeze(0);
      auto spec_feat_ret = spec_feat.unsqueeze(0).unsqueeze(0);

      return { spec_ret , erb_feat_ret, spec_feat_ret };
   }

   torch::Tensor DeepFilter::_analysis_time(torch::Tensor input_data)
   {
      auto in_chunks = torch::split(input_data, _frame_size);

      // time chunks iteration
      std::vector< torch::Tensor > output;
      for (auto ichunk : in_chunks)
      {
         output.push_back(_frame_analysis(ichunk));
      }

      return torch::stack(output, 0);
   }

   torch::Tensor DeepFilter::_frame_analysis(torch::Tensor input_frame)
   {
      //First part of the window on the previous frame
      //Second part of the window on the new input frame
      auto buf = torch::cat({ _reg_analysis_mem, input_frame }) * _window;
      auto buf_fft = torch::fft::rfft(buf, {}, -1, "backward") * _wnorm;

      // Copy input to analysis_mem for next iteration
      _reg_analysis_mem = input_frame;

      return buf_fft;
   }

   torch::Tensor DeepFilter::_erb_norm_time(torch::Tensor input_data, float alpha)
   {
      std::vector<torch::Tensor> output;

      for (int64_t i = 0; i < input_data.size(0); ++i) {
         torch::Tensor in_step = input_data[i];
         output.push_back(_band_mean_norm_erb(in_step, alpha));
      }

      return torch::stack(output, 0);
   }

   torch::Tensor DeepFilter::_band_mean_norm_erb(torch::Tensor xs, float alpha, float denominator)
   {
      _reg_erb_norm_state = torch::lerp(xs, _reg_erb_norm_state, alpha);
      auto output = (xs - _reg_erb_norm_state) / denominator;
      return output;
   }

   torch::Tensor DeepFilter::_erb(const torch::Tensor& input_data, bool db) {
      torch::Tensor magnitude_squared = torch::real(input_data).pow(2) + torch::imag(input_data).pow(2);
      torch::Tensor erb_features = (*_erb_conv)->forward(magnitude_squared.unsqueeze(-1)).squeeze(-1);

      if (db) {
         erb_features = 10.0 * torch::log10(erb_features + 1e-10);
      }

      return erb_features;
   }

   torch::Tensor DeepFilter::_unit_norm_time(torch::Tensor input_data, float alpha)
   {
      std::vector<torch::Tensor> output;

      for (int64_t i = 0; i < input_data.size(0); ++i) {
         torch::Tensor in_step = input_data[i];
         output.push_back(_band_unit_norm(in_step, alpha));
      }
      return torch::stack(output, 0);
   }

   torch::Tensor DeepFilter::_band_unit_norm(torch::Tensor xs, float alpha)
   {
      _reg_band_unit_norm_state = torch::lerp(xs.abs(), _reg_band_unit_norm_state, alpha);
      auto output = xs / _reg_band_unit_norm_state.sqrt();

      return output;
   }

   torch::Tensor DeepFilter::_synthesis_time(torch::Tensor input_data)
   {
      std::vector<torch::Tensor> out_chunks;
      for (int64_t i = 0; i < input_data.size(0); ++i) {
         torch::Tensor ichunk = input_data[i];
         auto output_frame = _frame_synthesis(ichunk);
         out_chunks.push_back(output_frame);
      }

      return torch::cat(out_chunks);
   }

   torch::Tensor DeepFilter::_frame_synthesis(torch::Tensor input_data)
   {
      auto x = torch::fft::irfft(input_data, {}, -1, "forward") * _window;

      auto split_ret = torch::split(x, { _frame_size, x.size(0) - _frame_size });
      auto x_first = split_ret[0];
      auto x_second = split_ret[1];

      auto output = x_first + _reg_synthesis_mem;

      _reg_synthesis_mem = x_second;

      return output;
   }
}
