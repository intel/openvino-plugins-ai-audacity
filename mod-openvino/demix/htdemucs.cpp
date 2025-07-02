// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "demix/htdemucs.h"
#include "musicgen/musicgen_utils.h"

namespace ov_demix
{
    HTDemucs::HTDemucs(const std::string& model_dir, const std::string& device, const std::string& cache_dir)
    {
        ov::Core core;

        auto xml_path = FullPath(model_dir, "htdemucs_fwd.xml");
        auto model = core.read_model(xml_path);
        std::cout << "htdemucs_v4.xml:" << std::endl;
        logBasicModelInfo(model);

        ov::AnyMap properties = { ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY) };

        if (!cache_dir.empty())
        {
            properties.insert(ov::cache_dir(cache_dir));
        }

        auto compiled_model = core.compile_model(model, device, properties);
        _forward_ir = compiled_model.create_infer_request();

        auto xt_out_shape = _forward_ir.get_tensor("xt_out").get_shape();
        if (xt_out_shape.size() != 3)
        {
            throw std::runtime_error("HTDemucs: Expected xt tensor to have rank 3, but it is "
                + std::to_string(xt_out_shape.size()));
        }
        _chunk_size = static_cast<int64_t>(xt_out_shape[2]);

        if (xt_out_shape[1] % 2 != 0)
        {
            throw std::runtime_error("HTDemucs: Expected xt tensor shape[1] to be divisible by 2. It is"
                + std::to_string(xt_out_shape[1]));
        }
        _num_instruments = xt_out_shape[1] / 2;

        std::cout << "HTDemucs: chunk size = " << _chunk_size << std::endl;
        std::cout << "HTDemucs: num instruments = " << _num_instruments << std::endl;
    }

    static void spectro(torch::Tensor& x, torch::Tensor& z, int n_fft, int hop_length, int pad = 0)
    {
        std::vector<int64_t> other = { x.sizes()[0], x.sizes()[1] };
        int64_t length = x.sizes()[2];

        x = x.reshape({ -1, length });

        //std::cout << "x.shape after x.reshape({ -1, length }); = " << x.sizes() << std::endl;

        torch::Tensor hwindow = torch::hann_window(n_fft).to(x);

        //z = torch::stft(x, n_fft * (1 + pad), hop_length, n_fft, hwindow, true, true, true, )
        z = torch::stft(x,
            n_fft * (1 + pad), //n_fft
            hop_length, //hop_length
            n_fft, //win_length
            hwindow, //window
            true, //center
            "reflect", //pad_mode
            true, //normalized
            true, //onesided
            true); //return_complex

        int64_t freqs = z.sizes()[1];
        int64_t frame = z.sizes()[2];

        z = z.view({ other[0], other[1], freqs, frame });
    }

    static void ispectro(torch::Tensor& z, torch::Tensor& x, int64_t hop_length, int64_t length, int64_t pad = 0)
    {
        std::vector<int64_t> other = { z.sizes()[0], z.sizes()[1], z.sizes()[2] };
        int64_t freqs = z.sizes()[3];
        int64_t frames = z.sizes()[4];
        int64_t n_fft = 2 * freqs - 2;
        z = z.view({ -1, freqs, frames });
        int64_t win_length = n_fft / (1 + pad);
        torch::Tensor hwindow = torch::hann_window(win_length).to(z.to(torch::kFloat32));
        x = torch::istft(z, n_fft, hop_length, win_length, hwindow, true, true, true, length);
        length = x.sizes()[1];
        x = x.view({ other[0], other[1], other[2], length });
    }

    static void pad1d(torch::Tensor& x, torch::Tensor& out, std::vector<int64_t> paddings, float val = 0.)
    {
        namespace F = torch::nn::functional;
        //Tiny wrapper around F.pad, just to allow for reflect padding on small input.
        // If this is the case, we insert extra 0 padding to the right before the reflection happen.
        auto length = x.sizes()[2];
        int64_t padding_left = paddings[0];
        int64_t padding_right = paddings[1];
        int max_pad = std::max(padding_left, padding_right);
        if (length <= max_pad)
        {
            int64_t extra_pad = max_pad - length + 1;
            int64_t extra_pad_right = std::min(padding_right, extra_pad);
            int64_t extra_pad_left = extra_pad - extra_pad_right;
            paddings = { padding_left - extra_pad_left, padding_right - extra_pad_right };
            x = F::pad(x, F::PadFuncOptions({ extra_pad_left, extra_pad_right }));
        }

        out = F::pad(x, F::PadFuncOptions(paddings).mode(torch::kReflect));
    }

    static void spec(torch::Tensor& x, torch::Tensor& z)
    {
        int64_t hl = 1024;
        int64_t nfft = 4096;

        // We re-pad the signal in order to keep the property
        // that the size of the output is exactly the size of the input
        // divided by the stride (here hop_length), when divisible.
        // This is achieved by padding by 1/4th of the kernel size (here nfft).
        // which is not supported by torch.stft.
        // Having all convolution operations follow this convention allow to easily
        // align the time and frequency branches later on.
        int64_t le = int64_t(std::ceil((float)x.sizes()[2] / (float)hl));
        int64_t pad = hl / 2 * 3;

        pad1d(x, x, { pad, pad + le * hl - x.sizes()[2] });

        //z = spectro(x, nfft, hl)[..., :-1, :]
        spectro(x, z, nfft, hl);
        z = z.index({ "...", torch::indexing::Slice(0, z.sizes()[2] - 1),  torch::indexing::Slice(torch::indexing::None) });

        //z = z[..., 2: 2 + le]
        z = z.index({ "...", torch::indexing::Slice(2, 2 + le) });
    }

    static void magnitude(torch::Tensor& z, torch::Tensor& m)
    {
        int64_t B = z.sizes()[0];
        int64_t C = z.sizes()[1];
        int64_t Fr = z.sizes()[2];
        int64_t T = z.sizes()[3];
        m = torch::view_as_real(z).permute({ 0, 1, 4, 2, 3 });
        m = m.reshape({ B, C * 2, Fr, T });
    }

    static void mask_no_z(torch::Tensor& m, torch::Tensor& out)
    {
        int64_t B = m.sizes()[0];
        int64_t S = m.sizes()[1];
        int64_t C = m.sizes()[2];
        int64_t Fr = m.sizes()[3];
        int64_t T = m.sizes()[4];
        out = m.view({ B, S, -1, 2, Fr, T }).permute({ 0, 1, 2, 4, 5, 3 });
        out = torch::view_as_complex(out.contiguous());
    }

    static void ispec(torch::Tensor& z, torch::Tensor& x, int64_t length)
    {
        namespace F = torch::nn::functional;
        int64_t hl = 1024;
        z = F::pad(z, F::PadFuncOptions({ 0, 0, 0, 1 }));
        z = F::pad(z, F::PadFuncOptions({ 2, 2 }));
        int64_t pad = hl / 2 * 3;
        int64_t le = hl * (int64_t)(std::ceil((float)length / (float)hl)) + 2 * pad;
        ispectro(z, x, hl, le);
        x = x.index({ "...", torch::indexing::Slice(pad, pad + length) });
    }

    torch::Tensor HTDemucs::run(torch::Tensor mix_tensor)
    {
        //length = mix.shape[-1]
        int64_t length = mix_tensor.sizes()[2];

        if (length != _chunk_size)
            throw std::runtime_error("Expected length to be chunk_size=" + std::to_string(_chunk_size)
                + ", but it is " + std::to_string(length));

        torch::Tensor mix = mix_tensor;

        torch::Tensor z;
        spec(mix_tensor, z);

        torch::Tensor mag;
        magnitude(z, mag);

        auto x = mag;
        int64_t B = x.sizes()[0];
        int64_t C = x.sizes()[1];
        int64_t Fq = x.sizes()[2];
        int64_t T = x.sizes()[3];

        //unlike previous Demucs, we always normalize because it is easier.
        torch::Tensor mean = x.mean({ 1, 2, 3 }, true);
        torch::Tensor std = x.std({ 1, 2, 3 }, true);

        // x will be the freq.branch input.

        // Prepare the time branch input.
        torch::Tensor xt = mix;
        torch::Tensor meant = xt.mean({ 1, 2 }, true);
        torch::Tensor stdt = xt.std({ 1, 2 }, true);
        xt = (xt - meant) / (1e-5 + stdt);

        x = (x - mean) / (1e-5 + std);

        {
            ov::Tensor x_tensor = _forward_ir.get_tensor("x");
            ov::Tensor xt_tensor = _forward_ir.get_tensor("xt");

            x.contiguous();
            xt.contiguous();

            auto pXTensor = x_tensor.data<float>();
            auto pXTTensor = xt_tensor.data<float>();

            std::memcpy(pXTensor, x.data_ptr(), x.numel() * x.element_size());
            std::memcpy(pXTTensor, xt.data_ptr(), xt.numel() * xt.element_size());

            _forward_ir.infer();

            ov::Tensor x_out_tensor = _forward_ir.get_tensor("x_out");
            ov::Tensor xt_out_tensor = _forward_ir.get_tensor("xt_out");

            auto pXTensor_Out = x_out_tensor.data<float>();
            auto pXTTensor_Out = xt_out_tensor.data<float>();

            auto x_as_tensor = wrap_ov_tensor_as_torch(x_out_tensor);
            auto xt_as_tensor = wrap_ov_tensor_as_torch(xt_out_tensor);
            x = x_as_tensor.clone();
            xt = xt_as_tensor.clone();
        }

        int64_t S = _num_instruments;
        x = x.view({ B, S, -1, Fq, T });

        //x = x * std[:, None] + mean[:, None]
        x = x * std.index({ torch::indexing::Slice(), torch::indexing::None }) + mean.index({ torch::indexing::Slice(), torch::indexing::None });

        torch::Tensor zout;
        mask_no_z(x, zout);

        ispec(zout, x, _chunk_size);

        xt = xt.view({ B, S, -1, _chunk_size });
        xt = xt * stdt.index({ torch::indexing::Slice(), torch::indexing::None }) + meant.index({ torch::indexing::Slice(), torch::indexing::None });
        x = xt + x;

        return x;
    }
}
