// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#include "musicgen_for_conditional_generation_refactor.h"

#include "encodec_encoder.h"
#include "musicgen_decoder.h"

namespace ov_musicgen
{
   void MusicgenForConditionalGenerationRefactor::SetSeed(unsigned int seed)
   {
      torch::Generator generator = at::detail::createCPUGenerator();
      {
         std::lock_guard<std::mutex> lock(generator.mutex());
         generator.set_current_seed(seed);
      }

      _generator = generator;
   }

   MusicgenForConditionalGenerationRefactor::MusicgenForConditionalGenerationRefactor(MusicGenConfig& config)
      : _config(config)
   {
      auto model_folder = config.model_folder;
      std::string cache_dir = *config.cache_folder;

      _core = std::make_shared< ov::Core >();
      auto& core = *_core;

      _decoder_refactor = std::make_shared< MusicgenDecoderStatic >(core, config);

      if (config.cache_folder)
      {
         std::cout << "Setting cache_dir to " << *config.cache_folder << std::endl;

         core.set_property(ov::cache_dir(*config.cache_folder));
      }

      torch::Generator generator = at::detail::createCPUGenerator();
      {
         std::lock_guard<std::mutex> lock(generator.mutex());
         generator.set_current_seed(1);
      }

      _generator = generator;

      {
         //prep text encoder
         auto modelpath = FullPath(model_folder, "t5.xml");
         std::shared_ptr<ov::Model> model = core.read_model(modelpath);

         //logBasicModelInfo(model);
         //TODO: Expose text encoder device?
         auto compiled_model = core.compile_model(model, "CPU");

         _text_encoder_infer_request = compiled_model.create_infer_request();
         _text_encoder_infer_request.infer();
      }


      {
         //prep enc-to-dec proj model
         std::string enc_to_dec_model_folder = model_folder;
         if (config.bStereo)
         {
            enc_to_dec_model_folder = FullPath(enc_to_dec_model_folder, "stereo");
         }
         else
         {
            enc_to_dec_model_folder = FullPath(enc_to_dec_model_folder, "mono");
         }

         auto modelpath = FullPath(enc_to_dec_model_folder, "enc_to_dec_proj.xml");
         std::shared_ptr<ov::Model> model = core.read_model(modelpath);

         model->reshape({ {2, ov::Dimension(1, 64), 768} });

         ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

         _enc_to_dec_proj_infer_request = compiled_model.create_infer_request();
      }

      {
         size_t num_encodec_secs = 20;

         auto modelpath = FullPath(model_folder, "encodec_" + std::to_string(num_encodec_secs) + "s.xml");
         auto binfile = FullPath(model_folder, "encodec_combined_weights.bin");

         std::shared_ptr<ov::Model> model = core.read_model(modelpath, binfile);

         model->reshape({ 1, 1, 4, num_encodec_secs * 50 });

         auto compiled_model = core.compile_model(model, config.encodec_dec_device);

         _encodec_infer_request = compiled_model.create_infer_request();
      }

      _encoder = std::make_shared< MusicGenEncodecEncoder >(core, config);
   }

   torch::Tensor MusicgenForConditionalGenerationRefactor::apply_delay_pattern_mask(torch::Tensor input_ids, torch::Tensor decoder_pad_token_mask)
   {
      using namespace torch::indexing;
      auto seq_len = input_ids.sizes().back();
      decoder_pad_token_mask = decoder_pad_token_mask.index({ "...", Slice(None, seq_len) });
      input_ids = torch::where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask);
      return input_ids;
   }

   torch::Tensor MusicgenForConditionalGenerationRefactor::prepare_inputs_for_generation(torch::Tensor decoder_input_ids, torch::Tensor decoder_delay_pattern_mask, std::optional<float> guidance_scale)
   {
      ITT_SCOPED_TASK(prepare_inputs_for_generation);
      using namespace torch::indexing;

      //apply delay pattern mask
      decoder_input_ids = apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask);

      if (guidance_scale && (*guidance_scale > 1))
      {
         decoder_input_ids = decoder_input_ids.repeat({ 2, 1 });
      }

      auto past_length = _decoder_refactor->PastLength();
      
      if (past_length >= 1)
      {
         int64_t remove_prefix_length;
         if (decoder_input_ids.sizes()[1] > past_length)
         {
            remove_prefix_length = past_length;
         }
         else
         {
            remove_prefix_length = decoder_input_ids.sizes()[1] - 1;
         }

         decoder_input_ids = decoder_input_ids.index({ Slice(), Slice(remove_prefix_length, None) });
      }

      return decoder_input_ids;

   }

   std::pair<torch::Tensor, torch::Tensor> MusicgenForConditionalGenerationRefactor::forward(std::optional<torch::Tensor> input_ids,
      std::optional<torch::Tensor> attention_mask,
      std::optional<torch::Tensor> input_values,
      std::optional<torch::Tensor> padding_mask,
      std::optional<torch::Tensor> decoder_input_ids,
      std::optional< BaseModelOutput > encoder_outputs,
      std::optional< torch::Tensor > encoder_hidden_states_in)
   {
      ITT_SCOPED_TASK(MusicgenForConditionalGeneration_forward)
      _nforward_calls++;

      torch::Tensor encoder_hidden_states;
      torch::Tensor encoder_attention_mask;

      if (encoder_outputs)
      {
         encoder_hidden_states = *(*encoder_outputs).last_hidden_state;

         if (attention_mask)
         {
            encoder_attention_mask = *attention_mask;
         }
         else
         {
            throw std::runtime_error("attention_mask is expected to be given with encoder_outputs");
         }
      }
      else
      {
         encoder_hidden_states = torch::zeros({ 2, 1, 768 });
         encoder_attention_mask = torch::zeros({ 2, 1 }, torch::dtype(torch::kInt64));
      }

      auto logits = _decoder_refactor->run(*decoder_input_ids,
         encoder_hidden_states,
         attention_mask);

      return { logits, encoder_hidden_states };
   }

   int64_t MusicgenForConditionalGenerationRefactor::MaxNewTokens()
   {
      return _decoder_refactor->MaxNewTokens();
   }

   void MusicgenForConditionalGenerationRefactor::ShiftLeft(int64_t ntokens)
   {

   }

   MusicgenForConditionalGenerationRefactor::GenerateReturn MusicgenForConditionalGenerationRefactor::generate(std::optional < torch::Tensor > inputs_tensor,
      int64_t max_token_length,
      std::optional < torch::Tensor > attention_mask,
      CallbackTracking& tracking,
      std::optional< torch::Tensor > audio_to_continue,
      std::optional< torch::Tensor > input_ids_to_continue,
      float guidance_scale,
      int64_t top_k,
      std::optional< CallbackParams > callback_params)
   {
      ITT_SCOPED_TASK(generate)

      GenerateReturn ret;

      if (max_token_length <= 0)
      {
         throw std::invalid_argument("max_token_length needs to be > 0");
      }

      _decoder_refactor->Reset();

      max_token_length += 4;

      int64_t pad_token_id = 2048;

      int64_t batch_size = 1;

      int64_t num_codebooks = 4;

      if (_config.bStereo)
      {
         num_codebooks = 8;
      }

      auto input_ids = torch::full({ num_codebooks, 1 }, pad_token_id, torch::kInt64);

      if (audio_to_continue)
      {
         torch::Tensor audio_codes;
         int64_t frames, bsz, codebooks, seq_len;
         if (!_config.bStereo)
         {
            if (audio_to_continue->size(1) != 1)
            {
               throw std::runtime_error("Models are configured for mono, but audio-to-continue != 1 channel");
            }

            audio_codes = _encoder->encode(*audio_to_continue);

            frames = audio_codes.size(0);
            bsz = audio_codes.size(1);
            codebooks = audio_codes.size(2);
            seq_len = audio_codes.size(3);

            if (frames != 1)
            {
               throw std::runtime_error("generate: expected frames to be 1");
            }
         }
         else 
         {
            if (audio_to_continue->size(1) != 2)
            {
               throw std::runtime_error("Models are configured for stereo, but audio-to-continue != 2 channels");
            }

            using namespace torch::indexing;

            auto input_vals_left = audio_to_continue->index({ Slice(), Slice(None, 1), Slice() });

            //careful! audio_codes_left is a thin wrapper around the encoder output tensor. 
            // this is why we do the index_put below before generating the right one (as it will implicitly)
            // change the values of audio_codes_left!
            auto audio_codes_left = _encoder->encode(input_vals_left);

            frames = audio_codes_left.size(0);
            bsz = audio_codes_left.size(1);
            codebooks = audio_codes_left.size(2);
            seq_len = audio_codes_left.size(3);

            // copy alternating left / right channel codes into stereo codebook
            audio_codes = audio_codes_left.new_ones({ frames, bsz, 2 * codebooks, seq_len });

            audio_codes.index_put_({ Slice(), Slice(), Slice(None, None, 2), Slice() }, audio_codes_left);

            auto input_vals_right = audio_to_continue->index({ Slice(), Slice(1, None), Slice() });

            auto audio_codes_right = _encoder->encode(input_vals_right);

            audio_codes.index_put_({ Slice(), Slice(), Slice(1, None, 2), Slice() }, audio_codes_right);
         }
         

         auto decoder_input_ids = audio_codes.index({ 0, "..." }).reshape({ bsz * num_codebooks , seq_len });

         decoder_input_ids = torch::cat({ input_ids, decoder_input_ids }, -1);

         input_ids = decoder_input_ids;
      }
      else if (input_ids_to_continue)
      {
         input_ids = torch::cat({ input_ids, *input_ids_to_continue }, -1);
      }

      //run text encoder
      std::optional< BaseModelOutput > encoder_outputs;
      if (inputs_tensor)
      {
         auto txt_encode_input = wrap_torch_tensor_as_ov(*inputs_tensor);
         _text_encoder_infer_request.set_input_tensor(txt_encode_input);
         _text_encoder_infer_request.infer();

         auto txt_encode_out = _text_encoder_infer_request.get_output_tensor();
         auto last_hidden_state = wrap_ov_tensor_as_torch(txt_encode_out);

         if (guidance_scale > 1)
         {
            last_hidden_state = torch::concatenate({ last_hidden_state, torch::zeros_like(last_hidden_state) }, 0);
         }

         BaseModelOutput output;
         output.last_hidden_state = last_hidden_state;

         encoder_outputs = output;
      }

      auto build_delay_pattern_mask_ret = _build_delay_pattern_mask(input_ids, pad_token_id, max_token_length);
      input_ids = build_delay_pattern_mask_ret.first;
      auto decoder_delay_pattern_mask = build_delay_pattern_mask_ret.second;

      auto output_ids = sample(input_ids,
         attention_mask,
         decoder_delay_pattern_mask,
         encoder_outputs,
         max_token_length,
         guidance_scale,
         top_k,
         tracking,
         callback_params);

      if (!output_ids.defined())
      {
         // this can happen in the event of user cancellation (via callback). In this case,
         // sample() will return an undefined torch::Tensor object.
         return ret;
      }

      ret.input_ids = output_ids;

      // apply the pattern mask to the final ids
      output_ids = apply_delay_pattern_mask(output_ids, decoder_delay_pattern_mask);

      // revert the pattern delay mask by filtering the pad token id
      torch::Tensor mask = output_ids != pad_token_id;
      output_ids = torch::masked_select(output_ids, mask);

      output_ids = torch::reshape(output_ids, { batch_size, num_codebooks, -1 });

      //  append the frame dimension back to the audio codes
      using namespace torch::indexing;
      output_ids = output_ids.index({ None, "..." });

      if (!_config.bStereo)
      {
         ret.wav = ids_to_wav(output_ids);
      }
      else
      {
         auto left_input = output_ids.index({ Slice(), Slice(), Slice(None, None, 2), Slice() });
         ret.wav = ids_to_wav(left_input);

         auto right_input = output_ids.index({ Slice(), Slice(), Slice(1, None, 2), Slice() });
         ret.wav1 = ids_to_wav(right_input);
      }

      return ret;
   }

   std::shared_ptr<std::vector<float>> MusicgenForConditionalGenerationRefactor::ids_to_wav(torch::Tensor ids)
   {
      using namespace torch::indexing;

      auto encodec_input_tensor = wrap_ov_tensor_as_torch(_encodec_infer_request.get_input_tensor());

      int64_t tokens_left_to_decode = ids.sizes()[3];

      //Conversion from number of tokens to number of audio samples is as follows..
      // There are 50 tokens per second
      // The EnCodec decoder produces 32 khz audio samples. (32000 samples for each second)
      // So, conversion is, (# of tokens)/ 50.0 * 32000.
      size_t number_of_output_samples = (size_t)std::ceil(((double)tokens_left_to_decode / 50.0) * 32000);

      std::shared_ptr<std::vector<float>> wav = std::make_shared< std::vector<float> >(number_of_output_samples);

      int64_t encodec_input_token_size = encodec_input_tensor.sizes()[3];
      int64_t num_tokens_decoded_so_far = 0;
      size_t num_samples_produced_so_far = 0;

      //if we are going to be calling encodec multiple times back to back, we want to establish
      // some overlap so that we don't produce any noticeable pops or clicks at the wav
      // concatenation point.
      size_t overlap_tokens = 5; //should be even
      if (tokens_left_to_decode <= encodec_input_token_size)
      {
         //if we can decode all tokens in a single pass then, we don't worry about overlap.
         overlap_tokens = 0;
      }

      size_t overlap_samples = (size_t)std::ceil(((double)overlap_tokens / 50.0) * 32000);

      //todo: EEK! This one got away from me a bit -- this code is a mess and is way more complicated than it has to be, but seems to be working
      // for now. Clean it up, and also rework it so that we can run multiple iterations of the loop in parallel
      // (as we can have multiple infer requests & use async API).
      size_t decodei = 0;
      while (tokens_left_to_decode > 0)
      {
         int64_t num_tokens_to_decode_this_it = std::min(encodec_input_token_size, tokens_left_to_decode);

         encodec_input_tensor.index_put_({ Slice(), Slice(), Slice(), Slice(0, num_tokens_to_decode_this_it) },
            ids.index({ Slice(), Slice(), Slice(), Slice(num_tokens_decoded_so_far,
                num_tokens_decoded_so_far + num_tokens_to_decode_this_it) }));

         //todo: if num_tokens_to_decode_this_it <= encodec_input_token_size, we should probably fill the remainder
         // tensor with some padding values...
         {
            ITT_SCOPED_TASK(encodec_decode)
               _encodec_infer_request.infer();
         }

         auto audio_values = _encodec_infer_request.get_output_tensor();

         size_t num_samples_this_it = (size_t)std::ceil(((double)num_tokens_to_decode_this_it / 50.0) * 32000);

         if (tokens_left_to_decode > encodec_input_token_size)
         {
            num_tokens_to_decode_this_it -= overlap_tokens;
            num_samples_this_it -= overlap_samples / 2;
         }

         size_t offset = 0;
         if (decodei > 0)
         {
            offset += overlap_samples / 2;
            num_samples_this_it -= overlap_samples / 2;
         }

         num_tokens_decoded_so_far += num_tokens_to_decode_this_it;
         tokens_left_to_decode -= num_tokens_to_decode_this_it;

         if ((num_samples_produced_so_far + num_samples_this_it) > wav->size())
         {
            throw std::runtime_error("Unexpectedly, the output wav vector doesn't have enough elements to hold decoded output samples.");
         }

         std::memcpy(wav->data() + num_samples_produced_so_far, audio_values.data<float>() + offset, num_samples_this_it * sizeof(float));

         num_samples_produced_so_far += num_samples_this_it;

         decodei++;
      }

      return wav;
   }

   torch::Tensor MusicgenForConditionalGenerationRefactor::sample(torch::Tensor input_ids,
      std::optional < torch::Tensor > attention_mask,
      torch::Tensor decoder_delay_pattern_mask,
      std::optional< BaseModelOutput > encoder_outputs,
      size_t max_length,
      float guidance_scale,
      int64_t top_k,
      CallbackTracking& tracking,
      std::optional< CallbackParams > callback_params)
   {
      ITT_SCOPED_TASK(MusicgenForConditionalGeneration_sample);

      using namespace std::chrono;
      using Clock = std::chrono::high_resolution_clock;

      uint64_t  t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

      std::optional< torch::Tensor > encoder_hidden_states;
      int64_t tokens_generated_count = -4;

      int nforward_calls = 0;
      while (true)
      {
         nforward_calls++;

         auto decoder_input_ids = prepare_inputs_for_generation(input_ids, decoder_delay_pattern_mask, guidance_scale);

         auto fwd_ret = forward(input_ids,
            attention_mask,
            {}, //input_values
            {}, //padding mask
            decoder_input_ids,
            encoder_outputs,
            encoder_hidden_states);

         auto logits = fwd_ret.first;
         encoder_hidden_states = fwd_ret.second;

         using namespace torch::indexing;
         auto next_token_logits = logits.index({ Slice(), -1, Slice() });

         auto next_token_scores = _logits_processor(input_ids, next_token_logits, guidance_scale);

         next_token_scores = _logits_warper(input_ids, next_token_scores, top_k, -INFINITY);

         torch::Tensor probs;
         {
            ITT_SCOPED_TASK(softmax)
               probs = torch::softmax(next_token_scores, -1);
         }

         torch::Tensor next_tokens;
         {
            ITT_SCOPED_TASK(multinomial)
               next_tokens = torch::multinomial(probs, 1, false, _generator).squeeze(1);
         }

         using namespace torch::indexing;

         input_ids = torch::cat({ input_ids, next_tokens.index({Slice(), None}) }, -1);

         tokens_generated_count++;

         if (tokens_generated_count > 0)
         {
            tracking.total_tokens_generated_so_far++;

            if (callback_params)
            {
               if ((tracking.total_tokens_generated_so_far % callback_params->every_n_new_tokens) == 0)
               {
                  if (callback_params->callback)
                  {
                     float perc = (float)tracking.total_tokens_generated_so_far / tracking.total_tokens_to_generate;
                     if (!callback_params->callback(perc, callback_params->user))
                     {
                        std::cout << "callback returned false. User cancelled." << std::endl;
                        //user cancelled.
                        return {};
                     }
                  }
               }
            }
         }

         if (input_ids.sizes()[1] >= max_length)
         {
            break;
         }
      }

      uint64_t  t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
      std::cout << "loop time = " << t1 - t0 << std::endl;

      return input_ids;
   }



   torch::Tensor MusicgenForConditionalGenerationRefactor::_enc_to_dec_proj(torch::Tensor encoder_hidden_states)
   {
      ITT_SCOPED_TASK(_enc_to_dec_proj)
         using namespace torch::indexing;

      auto ov_input_tensor = wrap_torch_tensor_as_ov(encoder_hidden_states);
      _enc_to_dec_proj_infer_request.set_input_tensor(ov_input_tensor);

      //run inference.
      _enc_to_dec_proj_infer_request.infer();

      //wrap output tensor as a torch tensor
      auto output_tensor_wrapped = wrap_ov_tensor_as_torch(_enc_to_dec_proj_infer_request.get_output_tensor());

      return output_tensor_wrapped;
   }

   torch::Tensor MusicgenForConditionalGenerationRefactor::_logits_processor(torch::Tensor input_ids, torch::Tensor next_token_logits, float guidance_scale)
   {
      ITT_SCOPED_TASK(_logits_processor)
         using namespace torch::indexing;

      size_t unguided_bsz = next_token_logits.sizes()[0] / 2;

      auto next_token_logits_split = next_token_logits.split(unguided_bsz, 0);
      auto cond_logits = next_token_logits_split[0];
      auto uncond_logits = next_token_logits_split[1];
      auto scores = uncond_logits + (cond_logits - uncond_logits) * guidance_scale;

      return scores;
   }

   torch::Tensor MusicgenForConditionalGenerationRefactor::_logits_warper(torch::Tensor input_ids, torch::Tensor next_token_scores, int64_t top_k, float filter_value)
   {
      ITT_SCOPED_TASK(_logits_warper)
         top_k = std::min(top_k, next_token_scores.sizes().back());

      auto topk_values = std::get<0>(torch::topk(next_token_scores, top_k));

      using namespace torch::indexing;
      auto selected = topk_values.index({ "...", -1, None });

      auto indices_to_remove = next_token_scores < selected;
      next_token_scores = next_token_scores.masked_fill(indices_to_remove, filter_value);
      return next_token_scores;
   }

   //returns { input_ids, delayed_pattern_mask }
   std::pair< torch::Tensor, torch::Tensor> MusicgenForConditionalGenerationRefactor::_build_delay_pattern_mask(torch::Tensor input_ids, int64_t pad_token_id, int64_t max_length)
   {
      using namespace torch::indexing;

      int64_t codebooks = 4;

      if (_config.bStereo)
      {
         codebooks = 8;
      }

      // (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
      input_ids = input_ids.reshape({ -1, codebooks, input_ids.sizes().back() });
      auto bsz = input_ids.sizes()[0];
      auto num_codebooks = input_ids.sizes()[1];
      auto seq_len = input_ids.sizes()[2];

      auto input_ids_shifted = (
         torch::ones({ bsz, num_codebooks, max_length }, torch::TensorOptions().dtype(torch::kInt64)) * -1
         );

      int64_t channel_codebooks;
      if (_config.bStereo)
      {
         channel_codebooks = num_codebooks / 2;
      }
      else
      {
         channel_codebooks = num_codebooks;
      }

      // we only apply the mask if we have a large enough seq len - otherwise we return as is
      if (max_length < (2 * channel_codebooks - 1))
      {
         return { input_ids.reshape({bsz * num_codebooks, -1}), input_ids_shifted.reshape({bsz * num_codebooks, -1}) };
      }

      // fill the shifted ids with the prompt entries, offset by the codebook idx
      for (int64_t codebook = 0; codebook < channel_codebooks; codebook++)
      {
         if (!_config.bStereo)
         {
            // mono channel - loop over the codebooks one-by-one
            input_ids_shifted.index_put_({ Slice(), codebook, Slice(codebook, seq_len + codebook) }, input_ids.index({ Slice(), codebook }));
         }
         else
         {
            // left/right channels are interleaved in the generated codebooks, so handle one then the other
            input_ids_shifted.index_put_({ Slice(), 2 * codebook, Slice(codebook, seq_len + codebook) }, input_ids.index({ Slice(), 2 * codebook }));
            input_ids_shifted.index_put_({ Slice(), 2 * codebook + 1, Slice(codebook, seq_len + codebook) }, input_ids.index({ Slice(), 2 * codebook + 1 }));
         }
      }

      // construct a pattern mask that indicates the positions of padding tokens for each codebook
      // first fill the upper triangular part (the EOS padding)
      auto delay_pattern = torch::triu(
         torch::ones({ channel_codebooks, max_length }, torch::TensorOptions().dtype(torch::kBool)), max_length - channel_codebooks + 1
      );

      // then fill the lower triangular part (the BOS padding)
      delay_pattern = delay_pattern + torch::tril(torch::ones({ channel_codebooks, max_length }, torch::TensorOptions().dtype(torch::kBool)));

      if (_config.bStereo)
      {
         // for left/right channel we need to duplicate every row of the pattern mask in an interleaved fashion
         delay_pattern = delay_pattern.repeat_interleave(2, 0);
      }

      auto mask = ~delay_pattern;
      input_ids = mask * input_ids_shifted + ~mask * pad_token_id;

      //dump_tensor(input_ids, "ov_input_ids_after_mask_op.raw");

      // find the first position to start generating - this is the first place we have the -1 token
      // and will always be in the first codebook (since it has no codebook offset)
      auto first_codebook_ids = input_ids.index({ Slice(), 0, Slice() });
      auto start_ids = (first_codebook_ids == -1).nonzero().index({ Slice(), 1 });

      int64_t first_start_id;
      if (start_ids.numel() > 0)
      {
         first_start_id = start_ids.min().item().toLong();
      }
      else
      {
         //we have no tokens that need to be filled - return entire matrix of input ids
         first_start_id = seq_len;
      }

      // (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
      auto pattern_mask = input_ids.reshape({ bsz * num_codebooks, -1 });

      input_ids = input_ids.index({ "...", Slice(None, first_start_id) }).reshape({ bsz * num_codebooks, -1 });

      return { input_ids , pattern_mask };
   }

}
