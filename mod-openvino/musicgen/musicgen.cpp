// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#include "musicgen.h"
#include <ittutils.h>

#define USE_REFACTORED 1
#if !USE_REFACTORED
#include "musicgen_for_conditional_generation.h"
#else
#include "musicgen_for_conditional_generation_refactor.h"
#endif

namespace ov_musicgen
{

   struct MusicGen::Impl
   {
      Impl(MusicGenConfig& config)
      {
#if USE_REFACTORED
         _gen = std::make_shared< MusicgenForConditionalGenerationRefactor >(config);
#else
         _gen = std::make_shared< MusicgenForConditionalGeneration >(config);
#endif

         ov::Core core;

#if OPENVINO_VERSION_MAJOR>=2024
#ifdef WIN32
         core.add_extension("openvino_tokenizers.dll");
#else
         core.add_extension("libopenvino_tokenizers.so");
#endif
#elif (OPENVINO_VERSION_MAJOR==2023)&&(OPENVINO_VERSION_MINOR==3)
#ifdef WIN32
         core.add_extension("user_ov_extensions.dll");
#else
         core.add_extension("libuser_ov_extensions.so");
#endif
#else
#error OpenVINO 2023.3.0 or later is required.
#endif

         auto tokenizer_model_path = FullPath(config.model_folder, "musicgen-small-tokenizer.xml");
         auto tokenizer_model = core.read_model(tokenizer_model_path);

         auto compiled_model = core.compile_model(tokenizer_model, "CPU");
         _tok_infer_request = compiled_model.create_infer_request();
      }

      ov::InferRequest _tok_infer_request;
#if USE_REFACTORED
      std::shared_ptr< MusicgenForConditionalGenerationRefactor > _gen;
#else
      std::shared_ptr< MusicgenForConditionalGeneration > _gen;
#endif
   };

   MusicGen::MusicGen(MusicGenConfig& config)
   {
      _config = config;

      _impl = std::make_shared< Impl >(_config);
   }

   static inline torch::Tensor pack_wav_to_tensor(std::shared_ptr<std::vector<float>> atc, int64_t ncontext_samples)
   {
      auto audio_to_continue_tensor = torch::zeros({ 1, ncontext_samples });

      int64_t atc_num_samples = atc->size();
      int64_t tensor_offset = 0;
      int64_t atc_offset = 0;
      int64_t size_to_copy = ncontext_samples;

      if (atc_num_samples < ncontext_samples)
      {
         tensor_offset = ncontext_samples - atc->size();
         size_to_copy = atc_num_samples;
      }
      else if (atc_num_samples > ncontext_samples)
      {
         atc_offset = atc_num_samples - ncontext_samples;
         size_to_copy = ncontext_samples;
      }

      memcpy((float*)(audio_to_continue_tensor.data_ptr()) + tensor_offset,
         atc->data() + atc_offset, size_to_copy * sizeof(float));

      return audio_to_continue_tensor;
   }

   std::pair<std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>>> MusicGen::Generate(std::optional<std::string> prompt,
      std::optional<AudioContinuationParams> audio_to_continue_params,
      float total_desired_length_seconds,
      std::optional< unsigned int > seed,
      float guidance_scale,
      int top_k,
      std::optional< CallbackParams > callback_params)
   {
      if (seed)
      {
         std::cout << "Setting seed of " << *seed << std::endl;
         _impl->_gen->SetSeed(*seed);
      }
      else
      {
         std::cout << "Seed not set. Defaulting to 1 " << std::endl;
      }

      std::optional< torch::Tensor > prompt_tokenized_ids;
      std::optional< torch::Tensor > attention_mask;
      std::vector<int64_t> ids_64;

      //if a prompt was passed, and it's not an empty string.
      if (prompt && !(prompt->empty()) && (*prompt != ""))
      {
         //tokenize
         {
            _impl->_tok_infer_request.set_input_tensor(ov::Tensor{ ov::element::string, {1}, &(*prompt) });
            _impl->_tok_infer_request.infer();

            auto input_ids = _impl->_tok_infer_request.get_tensor("input_ids");
            auto pInputIds = input_ids.data<int64_t>();
            for (size_t i = 0; i < input_ids.get_size(); i++)
            {
               ids_64.push_back(pInputIds[i]);
            }
         }

         //TODO: Rather than throwing an exception here. Reduce length to 64 (first 63 + manually appended '1'), print warning that we are running with truncated prompt.
         // (we can de-tokenize truncated id's to give exact prompt that was used).
         if (ids_64.size() > 64)
         {
            throw std::runtime_error("Given prompt is too long (it cannot exceed 64 tokens after tokenization)");
         }

         prompt_tokenized_ids = torch::from_blob(ids_64.data(), { 1, (int64_t)ids_64.size() }, torch::kInt64);
         attention_mask = torch::cat({ torch::ones(prompt_tokenized_ids->sizes()),  torch::zeros(prompt_tokenized_ids->sizes()) });
      }
      else
      {
         attention_mask = torch::zeros({ 2, 1 });
      }

      //generare will +4 to the tokens, so we need to subtract it here (yuck)
      const int64_t max_new_tokens_per_generate = _impl->_gen->MaxNewTokens() - 4;

      // 50 samples / sec
      int64_t total_tokens_left_to_generate = (int64_t)(std::ceil(total_desired_length_seconds * 50));
#if USE_REFACTORED
      MusicgenForConditionalGenerationRefactor::CallbackTracking tracking;
#else
      MusicgenForConditionalGeneration::CallbackTracking tracking;
#endif
      tracking.total_tokens_generated_so_far = 0;
      tracking.total_tokens_to_generate = total_tokens_left_to_generate;

      int64_t ncontext_samples;
      int64_t ncontext_tokens;
      switch (_config.m_continuation_context)
      {
      case MusicGenConfig::ContinuationContext::FIVE_SECONDS:
         ncontext_samples = 160000;
         ncontext_tokens = 250;
         break;

      case MusicGenConfig::ContinuationContext::TEN_SECONDS:
         ncontext_samples = 320000;
         ncontext_tokens = 500;
         break;
      }

      std::optional< torch::Tensor > audio_to_continue_tensor;
      size_t audio_to_continue_samples = 0;
      if (audio_to_continue_params)
      {
         auto audio_to_continue = audio_to_continue_params->audio_to_continue;

         if (!audio_to_continue.first)
         {
            throw std::invalid_argument("audio_to_continue_params were set, but audio_to_continue_params.audio_to_continue.first is not set.");
         }

         audio_to_continue_samples = audio_to_continue.first->size();

         audio_to_continue_tensor = pack_wav_to_tensor(audio_to_continue.first, ncontext_samples);

         if (audio_to_continue.second)
         {
            // double check that user set same number of 'right' samples as 'left'
            if (audio_to_continue.second->size() != audio_to_continue_samples)
            {
               throw std::invalid_argument("audio_to_continue_params.audio_to_continue.first->size() != ...second->size()!");
            }

            auto audio_to_continue_tensor1 = pack_wav_to_tensor(audio_to_continue.second, ncontext_samples);
            audio_to_continue_tensor = torch::stack({ audio_to_continue_tensor->squeeze(0), audio_to_continue_tensor1.squeeze() });

         }

         if (audio_to_continue_samples > ncontext_samples)
         {
            audio_to_continue_samples = ncontext_samples;
         }

         audio_to_continue_tensor = audio_to_continue_tensor->unsqueeze(0);
      }

      std::shared_ptr< std::vector<float> > output_wav0;
      std::shared_ptr< std::vector<float> > output_wav1;

      if (audio_to_continue_params && !audio_to_continue_params->bReturnAudioToContinueInOutput)
      {
         //this will cause us to *not* copy the re-encoded audio that was
         // passed in (although the user may want that in some modes).
         output_wav0 = std::make_shared<std::vector<float>>();

         if (_config.bStereo)
         {
            output_wav1 = std::make_shared<std::vector<float>>();
         }
      }

      size_t iterationi = 0;
      while (total_tokens_left_to_generate > 0)
      {
         int64_t max_new_tokens_we_can_generate_this_it = max_new_tokens_per_generate;
         int64_t context_tokens_this_it = 0;

         //if we are passing in some audio_to_continue, the max_length_this_iteration needs to include that.
         if (audio_to_continue_tensor)
         {
            max_new_tokens_we_can_generate_this_it -= ncontext_tokens;
            context_tokens_this_it = ncontext_tokens;
         }

         int64_t max_length_this_iteration = std::min(max_new_tokens_we_can_generate_this_it,
            total_tokens_left_to_generate);

         auto gen_ret = _impl->_gen->generate(prompt_tokenized_ids, max_length_this_iteration + context_tokens_this_it, attention_mask, tracking, audio_to_continue_tensor, {}, guidance_scale, top_k,
            callback_params);

         if (!gen_ret.wav)
         {
            //the user probably cancelled.
            return {};
         }

         if (!output_wav0)
         {
            // If we're doing audio continuation, and the user wants to receive the context as output, and we had to pad their input to fill the context.
            if (audio_to_continue_params && audio_to_continue_params->bReturnAudioToContinueInOutput && (ncontext_samples > audio_to_continue_samples))
            {
               output_wav0 = std::make_shared<std::vector<float>>();
               size_t offset = ncontext_samples - audio_to_continue_samples;

               auto wav = gen_ret.wav;
               output_wav0->insert(output_wav0->end(), wav->begin() + offset, wav->end());
            }
            else
            {
               output_wav0 = gen_ret.wav;
            }
         }
         else
         {
            auto wav = gen_ret.wav;

            //todo: crossfade between the start of the new clip and the end of the old clip.
            // Right now there are sometimes 'pops' or 'clicks' introduced at the point where
            // we merge these together.
            output_wav0->insert(output_wav0->end(), wav->begin() + ncontext_samples, wav->end());
         }

         if (gen_ret.wav1)
         {
            if (!output_wav1)
            {
               // If we're doing audio continuation, and the user wants to receive the context as output, and we had to pad their input to fill the context.
               if (audio_to_continue_params && audio_to_continue_params->bReturnAudioToContinueInOutput && (ncontext_samples > audio_to_continue_samples))
               {
                  output_wav1 = std::make_shared<std::vector<float>>();
                  size_t offset = ncontext_samples - audio_to_continue_samples;

                  auto wav = gen_ret.wav1;
                  output_wav1->insert(output_wav1->end(), wav->begin() + offset, wav->end());
               }
               else
               {
                  output_wav1 = gen_ret.wav1;
               }
            }
            else
            {
               auto wav = gen_ret.wav1;

               //todo: crossfade between the start of the new clip and the end of the old clip.
               // Right now there are sometimes 'pops' or 'clicks' introduced at the point where
               // we merge these together.
               output_wav1->insert(output_wav1->end(), wav->begin() + ncontext_samples, wav->end());
            }
         }

         total_tokens_left_to_generate -= max_length_this_iteration;

         //todo, handle stereo case.
         if (total_tokens_left_to_generate > 0)
         {
            audio_to_continue_tensor = torch::from_blob(output_wav0->data() + (output_wav0->size() - ncontext_samples), { 1, ncontext_samples });

            if (output_wav1)
            {
               auto audio_to_continue_tensor1 = torch::from_blob(output_wav1->data() + (output_wav1->size() - ncontext_samples), { 1, ncontext_samples });

               audio_to_continue_tensor = torch::stack({ audio_to_continue_tensor->squeeze(0), audio_to_continue_tensor1.squeeze() });
            }

            audio_to_continue_tensor = audio_to_continue_tensor->unsqueeze(0);
         }

         iterationi++;
      }

      return { output_wav0, output_wav1 };
   }
}
