#include "musicgen_model_static.h"
#include "music_gen_decoder_full_batch2.h"
#include "music_gen_decoder_full_batch1.h"
#include "music_gen_decoder_cl.h"

namespace ov_musicgen
{
   class MusicgenSinusoidalPositionalEmbedding
   {
   public:

      MusicgenSinusoidalPositionalEmbedding(int64_t num_positions, int64_t embedding_dim, MusicGenConfig& config)
         : _embedding_dim(embedding_dim)
      {
         //make_weights(num_positions, embedding_dim);

         std::string model_folder = config.model_folder;
         if (config.bStereo)
         {
            model_folder = FullPath(model_folder, "Stereo");
         }

         auto weights_file = FullPath(model_folder,
            "sinusoidal_positional_embedding_weights_" + std::to_string(num_positions) + "_" + std::to_string(embedding_dim) + ".raw");

         _emb_weights = read_tensor(weights_file, { num_positions, embedding_dim });
      }

      torch::Tensor forward(torch::Tensor input_ids, int64_t past_key_values_length)
      {
         ITT_SCOPED_TASK(MusicgenSinusoidalPositionalEmbedding_forward)
            auto bsz = input_ids.sizes()[0];
         auto codebooks = input_ids.sizes()[1];
         auto seq_len = input_ids.sizes()[2];

         // Create the position ids from the input token ids.
         auto position_ids = torch::arange(seq_len) + past_key_values_length;

         if (seq_len > _emb_weights.size(0))
         {
            throw std::runtime_error("MusicgenSinusoidalPositionalEmbedding: forward: unexpected case... seq_len > _emb_weights.size(0)");
         }

         return _emb_weights.index_select(0, position_ids.view(-1));
      }

   private:

      //this is actually unused right now. We can replicate the starting values of the nn.parameter weights with 
      // this function. But they change during training, so we actually just need to grab the values from the checkpoint,
      // which we've dumped as a .raw file.
      void make_weights(int64_t num_embeddings, int64_t embedding_dim)
      {
         _emb_weights = get_embedding(num_embeddings, embedding_dim);
      }

      torch::Tensor get_embedding(int64_t num_embeddings, int64_t embedding_dim)
      {
         auto half_dim = embedding_dim / 2;
         float emb_val = std::logf(10000.f) / (half_dim - 1);
         auto emb = torch::exp(torch::arange(half_dim) * -emb_val);
         emb = torch::arange(num_embeddings).unsqueeze(1) * emb.unsqueeze(0);
         emb = torch::cat({ torch::cos(emb), torch::sin(emb) }, 1).view({ num_embeddings, -1 });
         if (embedding_dim % 2 == 1)
         {
            //zero pad
            emb = torch::cat({ emb, torch::zeros({num_embeddings, 1}) }, 1);
         }

         return emb;
      }

      int64_t _embedding_dim;
      torch::Tensor _emb_weights;
   };

   MusicgenModelStatic::MusicgenModelStatic(ov::Core& core, MusicGenConfig& config)
   {
      auto model_folder = config.model_folder;
      auto cache_dir = config.cache_folder;

      if (config.bStereo)
      {
         _num_codebooks = 8;
      }
      else
      {
         _num_codebooks = 4;
      }

      bool bIsDev0GPU = config.musicgen_decode_device0.find("GPU") != std::string::npos;

      //if both devices specified are the same GPU device, use the CL-optimized variant.
      if ((config.musicgen_decode_device0 == config.musicgen_decode_device1) && bIsDev0GPU)
      {
         std::cout << "Using OpenCL-backed Decoder with device=" << config.musicgen_decode_device0 << std::endl;
         _decoder_model = std::make_shared< MusicgenDecoderModelCL >(core, config);
      }
      else
      {
         //If device0 and device1 match, use the batch2 variant -- unless the device is NPU, which doesn't support batch2 right now.
         if ((config.musicgen_decode_device0 == config.musicgen_decode_device1) && config.musicgen_decode_device0 != "NPU")
         {
            std::cout << "Using Batch2 Decoder with device=" << config.musicgen_decode_device0 << std::endl;
            _decoder_model = std::make_shared< MusicgenDecoderModelFullStaticBatch2 >(core, config);
         }
         else
         {
            std::cout << "Using Batch1 Decoder with devices=" << config.musicgen_decode_device0 << ", " << config.musicgen_decode_device1 << std::endl;
            std::cout << "               and initial device=" << config.initial_decode_device << std::endl;
            _decoder_model = std::make_shared< MusicgenDecoderModelFullStaticBatch1 >(core, config);
         }
      }


      //_decoder_model = std::make_shared< MusicgenDecoderModelFullStaticBatch1 >(core, model_folder, cache_dir);

      //creating MusicgenSinusoidalPositionalEmbedding instance with config.max_position_embeddings = 2048 config.hidden_size = 1024
      int64_t max_position_embeddings = 2048;
      int64_t hidden_size = 1024;
      _embed_positions = std::make_shared< MusicgenSinusoidalPositionalEmbedding >(max_position_embeddings, hidden_size, config);

      {

         std::string embed_tokens_model_folder = model_folder;
         if (config.bStereo)
         {
            embed_tokens_model_folder = FullPath(embed_tokens_model_folder, "Stereo");
         }

         auto modelpath = FullPath(embed_tokens_model_folder, "embed_tokens.xml");
         std::shared_ptr<ov::Model> model = core.read_model(modelpath);

         model->reshape({ {2, _num_codebooks, ov::Dimension()} });

         std::cout << "embed_tokens:" << std::endl;
         logBasicModelInfo(model);

         ov::CompiledModel emded_tokens_compiled_model = core.compile_model(model, "CPU");

         _embed_tokens_infer_request = emded_tokens_compiled_model.create_infer_request();
      }

      std::cout << "construction complete!" << std::endl;
   }

   ov::Tensor MusicgenModelStatic::forward(std::optional<torch::Tensor> input_ids,
      std::optional<torch::Tensor> attention_mask,
      std::optional<torch::Tensor> encoder_hidden_states,
      std::optional<torch::Tensor> encoder_attention_mask,
      std::optional<torch::Tensor> head_mask,
      std::optional<torch::Tensor> cross_attn_head_mask,
      std::optional<torch::Tensor> inputs_embeds)
   {
      ITT_SCOPED_TASK(MusicgenModelStatic_forward)

         int64_t past_length = _decoder_model->PastLength();

      torch::Tensor input;
      std::vector<int64_t> input_shape;

      if (input_ids)
      {
         input = torch::reshape(*input_ids, { -1, _num_codebooks , (*input_ids).sizes().back() });
         auto bsz = input.sizes()[0];
         auto seq_len = input.sizes()[2];

         input_shape = { bsz, seq_len };
      }
      else if (inputs_embeds)
      {
         throw std::runtime_error("not implemented yet.");
      }
      else
      {
         throw std::invalid_argument("You have to specify either input_ids or decoder_inputs_embeds");
      }

      //std::cout << "input_shape = " << input_shape << std::endl;
      if (!inputs_embeds)
      {
         inputs_embeds = _embed_tokens(input);
      }

      //todo?
      //attention_mask = _prepare_4d_causal_attention_mask(
      //    attention_mask, input_shape, inputs_embeds, past_key_values_length
      //)

      //std::cout << "encoder_attention_mask shape into prepare = " << encoder_attention_mask->sizes() << std::endl;
      if (encoder_hidden_states && encoder_attention_mask)
      {
         encoder_attention_mask = _prepare_4d_attention_mask(*encoder_attention_mask, input_shape.back());
      }

      //positions = self.decoder.embed_positions(input, past_key_values_length)
      auto positions = _embed_positions->forward(input, past_length);

      auto hidden_states = *inputs_embeds + positions;

      return _decoder_model->run(hidden_states, encoder_hidden_states, *encoder_attention_mask);
   }

   torch::Tensor MusicgenModelStatic::_prepare_4d_attention_mask(torch::Tensor mask, int64_t tgt_len)
   {
      using namespace torch::indexing;

      if (mask.sizes().size() != 2)
      {
         throw std::invalid_argument("_expand_mask: mask shape expected to have 2 dims");
      }

      auto bsz = mask.sizes()[0];
      auto src_len = mask.sizes()[1];

      auto expanded_mask = mask.index({ Slice(), None, None, Slice() }).expand({ bsz, 1, tgt_len, src_len }).toType(torch::kFloat);

      auto inverted_mask = 1.0f - expanded_mask;

      //on GPU, the following triggers an issue where all batch 1 results will be -nan
      //return inverted_mask.masked_fill(inverted_mask.toType(torch::kBool), -FLT_MAX);

      // this seem to fix it, but need to look into it. Not sure why -FLT_MAX is being 
      // set to an attention mask. Setting fill val as 0 produces same output, so... ??
      return inverted_mask.masked_fill(inverted_mask.toType(torch::kBool), 0);
   }

   torch::Tensor MusicgenModelStatic::_embed_tokens(torch::Tensor input)
   {
      ITT_SCOPED_TASK(_embed_tokens)
#if 0
         auto input_tensor_wrapped = wrap_ov_tensor_as_torch(_embed_tokens_infer_request.get_input_tensor());
      input_tensor_wrapped.copy_(input); //copy the contents of input into input_tensor_wrapped
#else
         auto ov_input = wrap_torch_tensor_as_ov(input);

      //note, the following 'dummy' nonsense is to work around some issue when we set 'ov_input'
      // directly as input_tensor. 
      ov::Tensor dummy_in = ov::Tensor(ov_input.get_element_type(), ov_input.get_shape());
      auto dummy = wrap_ov_tensor_as_torch(dummy_in);
      dummy.copy_(input);

      _embed_tokens_infer_request.set_input_tensor(dummy_in);
#endif
      _embed_tokens_infer_request.infer();

      auto out = wrap_ov_tensor_as_torch(_embed_tokens_infer_request.get_output_tensor());

      return out;
   }
}
