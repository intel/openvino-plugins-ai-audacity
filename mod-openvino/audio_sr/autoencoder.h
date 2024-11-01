#pragma once

#include "audiosr_common.h"

class DiagonalGaussianDistribution;
class AutoEncoder
{
public: 

    AutoEncoder(std::shared_ptr< AudioSR_Config > config);

    std::shared_ptr< DiagonalGaussianDistribution > encode(torch::Tensor x);

    torch::Tensor decode(torch::Tensor z);

    void set_config(AudioSR_Config config);

private:


    std::shared_ptr< AudioSR_Config > _config;

    ov::InferRequest _encoder_infer;
    ov::InferRequest _quant_conv_infer;

    ov::InferRequest _post_quant_conv;
    ov::InferRequest _decoder_infer;

    void _init_audiosr_encoder();
    void _init_quant_conv();
    void _init_post_quant_conv();
    void _init_audiosr_decoder();
};
