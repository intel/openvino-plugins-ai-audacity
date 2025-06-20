// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "demix/demix.h"
#include "musicgen/musicgen_utils.h"

namespace ov_demix
{
    static torch::Tensor getWindowingArray(int64_t window_size, int64_t fade_size)
    {
        // Create fade-in and fade-out linear ramps
        auto fadein = torch::linspace(0.0, 1.0, fade_size, torch::kFloat32);
        auto fadeout = torch::linspace(1.0, 0.0, fade_size, torch::kFloat32);

        // Initialize full window to 1s
        auto window = torch::ones({ window_size }, torch::kFloat32);

        // Apply fade-in and fade-out
        window.slice(0, 0, fade_size) = fadein;
        window.slice(0, window_size - fade_size, window_size) = fadeout;

        return window;
    }

    static torch::Tensor makeTensorFromTrack(const AudioTrack& track) {
        if (!track.first)
        {
            throw std::runtime_error("makeTensorFromTrack: track.first (left, or mono channel) is not set!");
        }

        if (track.first->empty())
        {
            throw std::runtime_error("makeTensorFromTrack: track.first (left, or mono channel) has no samples!");
        }

        int64_t n_channels = 1;
        if (track.second)
        {
            n_channels++;
            if (track.second->size() != track.first->size())
            {
                throw std::runtime_error("makeTensorFromTrack: mismatched number of samples between L & R channels");
            }
        }

        int64_t n_samples = track.first->size();

        // Allocate output tensor: shape [1, 2, n_samples]
        torch::Tensor output = torch::empty({ 2, n_samples }, torch::kFloat32);

        // Pointer to raw memory
        float* output_ptr = output.data_ptr<float>();

        for (int ch = 0; ch < 2; ++ch)
        {
            // Determine source channel
            const auto& src_vec = ((n_channels == 2) && (ch == 1)) ? track.second : track.first;

            // Copy to output tensor's appropriate slice
            std::memcpy(output_ptr + ch * n_samples, src_vec->data(), n_samples * sizeof(float));
        }

        return output;
    }

    static std::vector<AudioTrack> extractTracks(const torch::Tensor& tensor,
        const int64_t num_instruments,
        bool bReturnMono) {
        // Assert tensor shape is [1, 2, N]
        TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == num_instruments && tensor.size(1) == 2,
            "Expected shape [<num_instruments>, 2, N]");

        // Ensure the tensor is contiguous before we copy from it.
        auto copy_from_tensor = tensor.contiguous();

        int64_t num_channels = bReturnMono ? 1 : 2;
        int64_t num_samples = tensor.size(2);

        std::vector<AudioTrack> result;
        for (int64_t instrumenti = 0; instrumenti < num_instruments; instrumenti++)
        {
            AudioTrack track;

            //left
            {
                track.first = std::make_shared<std::vector<float>>(num_samples);
                torch::Tensor ch_tensor = copy_from_tensor[instrumenti][0];
                std::memcpy(track.first->data(), ch_tensor.data_ptr<float>(), num_samples * sizeof(float));
            }

            //right
            if (!bReturnMono)
            {
                track.second = std::make_shared<std::vector<float>>(num_samples);
                torch::Tensor ch_tensor = copy_from_tensor[instrumenti][1];
                std::memcpy(track.second->data(), ch_tensor.data_ptr<float>(), num_samples * sizeof(float));
            }

            result.emplace_back(track);
        }

        return result;
    }


    static inline torch::Tensor pad(torch::Tensor t, int64_t left, int64_t right, DemixModel::PadMode pad_mode)
    {
        switch (pad_mode)
        {
            case DemixModel::PadMode::Constant0:
                t = torch::constant_pad_nd(t, { left, right }, 0);
                break;

            case DemixModel::PadMode::Reflect:
                t = torch::nn::functional::pad(
                    t,
                    torch::nn::functional::PadFuncOptions({ left, right }).mode(torch::kReflect)
                );
                break;

            default:
                throw std::runtime_error("Unsupported pad mode: " + std::to_string((int)pad_mode));
        }

        return t;
    }

    // This function is a C++ port of the 'demix' function found within ZFTurbo's excellent 'Music-Source-Separation-Training' project.
    // Here's a link to the implementation that was ported: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/a8a862231de29e6832a26c590e5c29f82f736f66/utils/model_utils.py#L15
    // Note that there are some minor tweaks...
    std::vector<AudioTrack> Demix(std::shared_ptr<DemixModel> model,
        const AudioTrack& track,
        const int n_overlap,
        ProgressUpdate fn,
        void* progress_update_user)
    {
        if (!model)
        {
            throw std::runtime_error("model is null!");
        }

        auto mix = makeTensorFromTrack(track);

        bool bMonoInput = !track.second;

        auto chunk_size = model->chunk_size();
        int64_t num_overlap = n_overlap;
        auto fade_size = chunk_size / 10;
        auto step = chunk_size / num_overlap;
        auto border = chunk_size - step;
        auto length_init = mix.size(-1);

        torch::Tensor windowing_array;

        auto pad_mode = model->pad_mode();

        if (model->PadOuterAndCrossFade())
        {
            windowing_array = getWindowingArray(chunk_size, fade_size);
            if (length_init > 2 * border && border > 0)
            {
                mix = pad(mix, border, border, pad_mode);
            }
        }

        size_t batch_size = 1;
        auto num_instruments = model->num_instruments();
        std::vector<int64_t> req_shape = { num_instruments };
        for (auto& s : mix.sizes())
        {
            req_shape.push_back(s);
        }

        auto result = torch::zeros(req_shape, torch::dtype(torch::kFloat32));
        auto counter = torch::zeros(req_shape, torch::dtype(torch::kFloat32));

        for (int i = 0; i < mix.size(1); i += step)
        {
            // part = mix[:, i:i + chunk_size]
            auto part = mix.index({ torch::indexing::Slice(), torch::indexing::Slice(i, i + chunk_size) });

            auto chunk_len = part.size(-1);
            auto part_pad_mode = pad_mode;
            if (pad_mode == DemixModel::PadMode::Reflect && chunk_len <= chunk_size / 2)
            {
                part_pad_mode = DemixModel::PadMode::Constant0;
            }
            part = pad(part, 0, chunk_size - chunk_len, part_pad_mode);

            auto arr = part.unsqueeze(0);

            arr = arr.contiguous();
            auto x = model->run(arr);

            int64_t start = i;
            int64_t seg_len = chunk_len;
            if (model->PadOuterAndCrossFade())
            {
                auto window = windowing_array.clone();
                if (i - step == 0) //First audio chunk, no fadein
                {
                    window.index({ torch::indexing::Slice(0, fade_size) }).fill_(1);
                }
                else if (i >= mix.size(1)) // Last audio chunk, no fadeout
                {
                    window.index({ torch::indexing::Slice(-fade_size, torch::indexing::None) }).fill_(1);
                }

                //result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                //counter[..., start:start + seg_len] += window[..., :seg_len]
                auto result_slice = result.index({ torch::indexing::Ellipsis, torch::indexing::Slice::Slice(start, start + seg_len) });
                auto x_slice = x.index({ 0, torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, seg_len) });
                auto counter_slice = counter.index({ torch::indexing::Ellipsis, torch::indexing::Slice(start, start + seg_len) });
                auto window_slice = window.index({ torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, seg_len) });
                result_slice.add_(x_slice * window_slice);
                counter_slice.add_(window_slice);
            }
            else
            {
                //result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu()
                //counter[..., start:start + seg_len] += 1.0
                auto result_slice = result.index({ torch::indexing::Ellipsis, torch::indexing::Slice::Slice(start, start + seg_len) });
                auto x_slice = x.index({ 0, torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, seg_len) });
                auto counter_slice = counter.index({ torch::indexing::Ellipsis, torch::indexing::Slice(start, start + seg_len) });
                result_slice.add_(x_slice);
                counter_slice.add_(1);
            }

            if (fn)
            {
                double perc_complete = (double)(i + step) / (double)mix.size(1);
                if (!fn(perc_complete, progress_update_user))
                {
                    std::cout << "cancelled." << std::endl;
                    return {};
                }
            }
        }

        auto estimated_sources = result / counter;
        estimated_sources = torch::nan_to_num(estimated_sources, /*nan=*/0.0);

        if (model->PadOuterAndCrossFade())
        {
            //remove the border that we added
            if (length_init > 2 * border && border > 0)
            {
                estimated_sources = estimated_sources.index({ torch::indexing::Ellipsis, torch::indexing::Slice(border, -border) });
            }
        }

        auto ret_tracks = extractTracks(estimated_sources, num_instruments, bMonoInput);
        return ret_tracks;
    }

    void GenerateInstrumental(const AudioTrack& stem_track, AudioTrack& input_output)
    {
        if (!stem_track.first)
        {
            throw std::runtime_error("GenerateInstrumental: stem_track track is missing first channel");
        }

        if (!input_output.first)
        {
            throw std::runtime_error("GenerateInstrumental: input_output track is missing first channel");
        }

        if (stem_track.second && !input_output.second)
        {
            throw std::runtime_error("GenerateInstrumental: input_output track is missing second channel");
        }

        if (stem_track.first->empty())
        {
            throw std::runtime_error("GenerateInstrumental: No samples in stem track, first channel");
        }

        if (stem_track.first->size() != input_output.first->size())
        {
            throw std::runtime_error("GenerateInstrumental: mismatched number of samples in first channel");
        }

        {
            auto* pStem = stem_track.first->data();
            auto* pInputOutput = input_output.first->data();
            for (int i = 0; i < stem_track.first->size(); i++)
            {
                pInputOutput[i] = pInputOutput[i] - pStem[i];
            }
        }

        if (stem_track.second)
        {
            if (stem_track.second->empty())
            {
                throw std::runtime_error("GenerateInstrumental: No samples in stem track, second channel");
            }

            if (stem_track.second->size() != input_output.second->size())
            {
                throw std::runtime_error("GenerateInstrumental: mismatched number of samples in second channel");
            }

            auto* pStem = stem_track.second->data();
            auto* pInputOutput = input_output.second->data();
            for (int i = 0; i < stem_track.second->size(); i++)
            {
                pInputOutput[i] = pInputOutput[i] - pStem[i];
            }
        }
    }
}
