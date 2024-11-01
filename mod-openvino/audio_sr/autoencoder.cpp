#include "autoencoder.h"
#include "distributions.h"

#define MODEL_FILE_EXT ".xml"



void AutoEncoder::_init_audiosr_encoder()
{
   auto& core = _config->core;

   auto modelpath = FullPath(_config->model_folder, "audiosr_encoder" MODEL_FILE_EXT);
   std::shared_ptr<ov::Model> model = core.read_model(modelpath);
   logBasicModelInfo(model);
   auto compiledModel = core.compile_model(model, _config->first_stage_encoder_device);
   _encoder_infer = compiledModel.create_infer_request();
}

void AutoEncoder::_init_quant_conv()
{
   auto& core = _config->core;

   auto modelpath = FullPath(_config->model_folder, "quant_conv" MODEL_FILE_EXT);
   std::shared_ptr<ov::Model> model = core.read_model(modelpath);
   logBasicModelInfo(model);
   auto compiledModel = core.compile_model(model, _config->first_stage_encoder_device);
   _quant_conv_infer = compiledModel.create_infer_request();

   //link encoder infer to quant conv 
   _quant_conv_infer.set_tensor("h", _encoder_infer.get_tensor("h"));
}

void AutoEncoder::_init_post_quant_conv()
{
   auto& core = _config->core;

   auto modelpath = FullPath(_config->model_folder, "post_quant_conv" MODEL_FILE_EXT);
   std::shared_ptr<ov::Model> model = core.read_model(modelpath);
   logBasicModelInfo(model);
   auto compiledModel = core.compile_model(model, _config->first_stage_encoder_device);
   _post_quant_conv = compiledModel.create_infer_request();
}

void AutoEncoder::_init_audiosr_decoder()
{
   auto& core = _config->core;

   auto modelpath = FullPath(_config->model_folder, "audiosr_decoder" MODEL_FILE_EXT);
   std::shared_ptr<ov::Model> model = core.read_model(modelpath);
   logBasicModelInfo(model);
   auto compiledModel = core.compile_model(model, _config->first_stage_encoder_device);
   _decoder_infer = compiledModel.create_infer_request();

   //link encoder infer to quant conv 
   _decoder_infer.set_input_tensor(_post_quant_conv.get_output_tensor());
}

AutoEncoder::AutoEncoder(std::shared_ptr< AudioSR_Config > config)
    : _config(config)
{
    if (!_config)
        throw std::runtime_error("AutoEncoder: config is null!");

    _init_audiosr_encoder();
    _init_quant_conv();
    _init_post_quant_conv();
    _init_audiosr_decoder();
}



void AutoEncoder::set_config(AudioSR_Config config)
{
   if (!_config)
      throw std::runtime_error("AutoEncoder::set_config: config is null!");

   if (config.first_stage_encoder_device != _config->first_stage_encoder_device)
   {
      _config->first_stage_encoder_device = config.first_stage_encoder_device;

      _init_audiosr_encoder();
      _init_quant_conv();
      _init_post_quant_conv();
      _init_audiosr_decoder();
   }
}


std::shared_ptr< DiagonalGaussianDistribution > AutoEncoder::encode(torch::Tensor x)
{
    auto x_ov_tensor = wrap_torch_tensor_as_ov(x);

    //std::cout << "x_ov_tensor shape = " << x_ov_tensor.get_shape() << std::endl;

    //run encoder
    _encoder_infer.set_tensor("x", x_ov_tensor);
    _encoder_infer.infer();

    //run quant_conv
    _quant_conv_infer.infer();

    auto moments_ov_tensor = _quant_conv_infer.get_tensor("moments");

    //std::cout << "moments_ov_tensor shape = " << moments_ov_tensor.get_shape() << std::endl;

    auto moments = wrap_ov_tensor_as_torch(moments_ov_tensor).clone();

    //std::cout << "Saving moments_ov.raw, which has shape = " << moments.sizes() << std::endl;
    //dump_tensor(moments, "moments_ov.raw");

    return std::make_shared< DiagonalGaussianDistribution >(moments);
}

torch::Tensor AutoEncoder::decode(torch::Tensor z)
{
    auto z_ov_tensor = wrap_torch_tensor_as_ov(z.contiguous());
    _post_quant_conv.set_input_tensor(z_ov_tensor);

    //run post-quant
    _post_quant_conv.infer();

    //run decode
    _decoder_infer.infer();

    auto dec = wrap_ov_tensor_as_torch(_decoder_infer.get_output_tensor()).clone();

    return dec;
}
