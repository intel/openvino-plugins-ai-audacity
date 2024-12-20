// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#include <torch/torch.h>

#include "htdemucs.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <openvino/openvino.hpp>

#ifdef ONNX_SUPPORT
#include <onnxruntime_cxx_api.h>
#endif

namespace ovdemucs
{
#ifdef ONNX_SUPPORT
    struct HTDemucs_impl
    {
        HTDemucs_impl(const char* model_path)
        {
            const size_t cSize = strlen(model_path) + 1;
            wchar_t* model_path_w = new wchar_t[cSize];
            mbstowcs(model_path_w, model_path, cSize);

            //set up onnx session.
            _ort_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "demucs4");

            const auto& api = Ort::GetApi();
            _session_options.SetIntraOpNumThreads(0);

            _session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
            _session = std::make_shared< Ort::Session >(_ort_env, model_path_w, _session_options);
        
            delete[] model_path_w;

            const size_t num_input_nodes = _session->GetInputCount();
            _input_names_ptr.reserve(num_input_nodes);
            _input_node_names.reserve(num_input_nodes);

            for (size_t i = 0; i < num_input_nodes; i++) {
                // print input node names
                auto input_name = _session->GetInputNameAllocated(i, _allocator);
                std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
                _input_node_names.push_back(input_name.get());
                _input_names_ptr.push_back(std::move(input_name));

                // print input node types
                auto type_info = _session->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                ONNXTensorElementDataType type = tensor_info.GetElementType();
                std::cout << "Input " << i << " : type = " << type << std::endl;

                // print input shapes/dims
                _input_node_dims.push_back(tensor_info.GetShape());
                std::cout << "Input " << i << " : num_dims = " << _input_node_dims[i].size() << '\n';
                for (size_t j = 0; j < _input_node_dims[i].size(); j++) {
                    std::cout << "Input " << i << " : dim[" << j << "] =" << _input_node_dims[i][j] << '\n';
                }
                std::cout << std::flush;
            }

            const size_t num_output_nodes = _session->GetOutputCount();
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = _session->GetOutputNameAllocated(i, _allocator);
                std::cout << "Output " << i << " : name =" << output_name.get() << std::endl;
                _output_node_names.push_back(output_name.get());
                _output_names_ptr.push_back(std::move(output_name));
            }
        }

        void run_inference(torch::Tensor& x, torch::Tensor& xt)
        {
            auto memory_info_x = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto memory_info_xt = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            std::vector<int64_t> x_dims;
            for (auto d : x.sizes())
                x_dims.push_back(d);

            std::vector<Ort::Value> inputTensors;

            x.contiguous();
            xt.contiguous();

            //"input.25"
            inputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info_x, (float*)x.data_ptr(), x.numel(),
                x_dims.data(), x_dims.size()));

            std::vector<int64_t> xt_dims;
            for (auto d : xt.sizes())
                xt_dims.push_back(d);

            //"input.1"
            inputTensors.push_back(Ort::Value::CreateTensor<float>(memory_info_xt, (float*)xt.data_ptr(), xt.numel(),
                xt_dims.data(), xt_dims.size()));

            std::cout << "Running ONNX Inference.." << std::endl;
            //std::cout << "_input_node_names.size() = " << _input_node_names.size() << std::endl;
            //std::cout << "_input_node_names = " << _input_node_names << std::endl;
            //std::cout << "_output_node_names.size() = " << _input_node_names.size() << std::endl;
            //std::cout << "_output_node_names = " << _output_node_names << std::endl;

            auto output_tensors =
                _session->Run(Ort::RunOptions{ nullptr }, _input_node_names.data(), inputTensors.data(), inputTensors.size(), _output_node_names.data(), _output_node_names.size());
            std::cout << "Running ONNX Inference.. DONE!" << std::endl;

            for (int i = 0; i < output_tensors.size(); i++)
            {
                //std::cout << "output " << i << ":" << std::endl;
                auto type_info = output_tensors[i].GetTypeInfo();
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                std::vector<int64_t> shape = tensor_info.GetShape();
                //std::cout << "    shape = " << shape << std::endl;
            }

            float* pX_Out = output_tensors[0].GetTensorMutableData<float>();
            float* pXt_Out = output_tensors[1].GetTensorMutableData<float>();

            torch::Tensor x_as_tensor = torch::from_blob(pX_Out, { 1, 16, 2048, 336 });
            torch::Tensor xt_as_tensor = torch::from_blob(pXt_Out, { 1, 8, 343980 });
            x = x_as_tensor.clone();
            xt = xt_as_tensor.clone();
        }

    private:

        Ort::Env _ort_env;
        Ort::SessionOptions _session_options;
        std::shared_ptr< Ort::Session > _session;
        
        Ort::AllocatorWithDefaultOptions _allocator;

        std::vector<const char*> _input_node_names;
        std::vector<Ort::AllocatedStringPtr> _input_names_ptr;
        std::vector<std::vector<int64_t>> _input_node_dims;

        std::vector<const char*> _output_node_names;
        std::vector<Ort::AllocatedStringPtr> _output_names_ptr;


    };
#endif

    static inline void logBasicModelInfo(const std::shared_ptr<ov::Model>& model) {
        std::cout << "Model name: " << model->get_friendly_name() << std::endl;

        // Dump information about model inputs/outputs
        ov::OutputVector inputs = model->inputs();
        ov::OutputVector outputs = model->outputs();

        std::cout << "\tInputs: " << std::endl;
        for (const ov::Output<ov::Node>& input : inputs) {
            const std::string name = input.get_any_name();
            const ov::element::Type type = input.get_element_type();
            const ov::PartialShape shape = input.get_partial_shape();
            const ov::Layout layout = ov::layout::get_layout(input);

            std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
        }

        std::cout << "\tOutputs: " << std::endl;
        for (const ov::Output<ov::Node>& output : outputs) {
            const std::string name = output.get_any_name();
            const ov::element::Type type = output.get_element_type();
            const ov::PartialShape shape = output.get_partial_shape();
            const ov::Layout layout = ov::layout::get_layout(output);

            std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
        }

        return;
    }

    static inline std::string fileExt(const std::string& filename) {
        auto pos = filename.rfind('.');
        if (pos == std::string::npos)
            return "";
        return filename.substr(pos + 1);
    }

    static inline void logCompiledModelInfo(const ov::CompiledModel& model) {
        //std::cout << "Model name: " << model.get_friendly_name() << std::endl;

        // Dump information about model inputs/outputs
        auto inputs = model.inputs();
        auto outputs = model.outputs();

        std::cout << "\tInputs: " << std::endl;
        for (const ov::Output<const ov::Node>& input : inputs) {
            const std::string name = input.get_any_name();
            const ov::element::Type type = input.get_element_type();
            const ov::PartialShape shape = input.get_partial_shape();
            const ov::Layout layout = ov::layout::get_layout(input);

            std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
        }

        std::cout << "\tOutputs: " << std::endl;
        for (const ov::Output<const ov::Node>& output : outputs) {
            const std::string name = output.get_any_name();
            const ov::element::Type type = output.get_element_type();
            const ov::PartialShape shape = output.get_partial_shape();
            const ov::Layout layout = ov::layout::get_layout(output);

            std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
        }

        return;
    }

    struct HTDemucs_openvino_impl
    {
        HTDemucs_openvino_impl(const char* model_path, const std::string& device = "CPU", const std::string cache_dir = "")
        {
            ov::Core core;

            if (!cache_dir.empty())
            {
               std::cout << "Setting cache_dir to " << cache_dir << std::endl;
               core.set_property(ov::cache_dir(cache_dir));
            }
            else
            {
               std::cout << "NOT Setting cache_dir"  << std::endl;
            }

            if (fileExt(model_path) == "blob")
            {
                std::ifstream modelStream(model_path, std::ios_base::binary | std::ios_base::in);
                if (!modelStream.is_open()) {
                    throw std::runtime_error("Cannot open model file " + std::string(model_path));
                }

                std::cout << "Importing pre-compiled blob" << std::endl;
                compiledModel = core.import_model(modelStream, device, {});
                std::cout << "Import complete." << std::endl;
                modelStream.close(); 
            }
            else
            {
                compiledModel = core.compile_model(model_path, device);
            }

            logCompiledModelInfo(compiledModel);

            const auto inputs = compiledModel.inputs();
            const auto outputs = compiledModel.outputs();
            for (const ov::Output<const ov::Node>& input : inputs) {
                const std::string name = input.get_any_name();
                inputsNames.push_back(name);
            }

            for (const ov::Output<const ov::Node>& output : outputs) {
                const std::string name = output.get_any_name();
                outputsNames.push_back(name);
            }

            inferRequest = compiledModel.create_infer_request();
        }

        void run_inference(torch::Tensor& x, torch::Tensor& xt)
        {
            const ov::Tensor& x_tensor = inferRequest.get_tensor(inputsNames[0]);
            const ov::Tensor& xt_tensor = inferRequest.get_tensor(inputsNames[1]);

            x.contiguous();
            xt.contiguous();

            float* pXTensor = x_tensor.data<float>();
            float* pXTTensor = xt_tensor.data<float>();

            std::memcpy(pXTensor, x.data_ptr(), x.numel() * x.element_size());
            std::memcpy(pXTTensor, xt.data_ptr(), xt.numel() * xt.element_size());

            static int inferencei = 0;
            inferRequest.infer();

            const ov::Tensor& x_out_tensor = inferRequest.get_tensor(outputsNames[0]);
            const ov::Tensor& xt_out_tensor = inferRequest.get_tensor(outputsNames[1]);

            float* pXTensor_Out = x_out_tensor.data<float>();
            float* pXTTensor_Out = xt_out_tensor.data<float>();

            torch::Tensor x_as_tensor = torch::from_blob(pXTensor_Out, { 1, 16, 2048, 336 });
            torch::Tensor xt_as_tensor = torch::from_blob(pXTTensor_Out, { 1, 8, 343980 });
            x = x_as_tensor.clone();
            xt = xt_as_tensor.clone();
        }

    private:

        std::vector<std::string> inputsNames;
        std::vector<std::string> outputsNames;

        ov::InferRequest inferRequest;
        ov::CompiledModel compiledModel;

    };

    std::vector<std::string> HTDemucs::GetSupportedDevices()
    {
       std::vector<std::string> device_list;
       ov::Core core;
       auto devices = core.get_available_devices();

       for (auto d : devices)
       {
          //GNA devices are not supported
          if (d.find("GNA") != std::string::npos) continue;

          device_list.push_back(d);
       }

       return device_list;
    }

    class TensorChunk
    {
    public:
        TensorChunk(torch::Tensor& t, int64_t offset = 0, int64_t length = -1)
        {
            int64_t total_length = t.sizes()[t.sizes().size() - 1];

            if (length < 0)
                length = total_length - offset;
            else
                length = std::min(total_length - offset, length);

            _tensor = t;
            _offset = offset;
            _length = length;
        }

        TensorChunk(TensorChunk& tc, int64_t offset = 0, int64_t length = -1)
        {
            int64_t total_length = tc.shape()[tc.shape().size() - 1];

            if (length < 0)
                length = total_length - offset;
            else
                length = std::min(total_length - offset, length);

            _tensor = tc._tensor;
            _offset = offset + tc._offset;
            _length = length;
        }

        std::vector<int64_t> shape()
        {
            std::vector<int64_t> s;// = _tensor.sizes();
            for (auto d : _tensor.sizes())
            {
                s.push_back(d);
            }

            s[s.size() - 1] = _length;

            return s;
        }


        void padded(int64_t target_length, torch::Tensor& out)
        {
            int64_t delta = target_length - _length;
            int64_t total_length = _tensor.sizes()[_tensor.sizes().size() - 1];

            int64_t start = _offset - delta / 2;
            int64_t end = start + target_length;

            int64_t correct_start = std::max((int64_t)0, start);
            int64_t correct_end = std::min(total_length, end);

            int64_t pad_left = correct_start - start;
            int64_t pad_right = end - correct_end;

            namespace F = torch::nn::functional;
            out = F::pad(_tensor.index({ "...", torch::indexing::Slice(correct_start, correct_end) }), F::PadFuncOptions({ pad_left, pad_right }));
        }

    private:

        torch::Tensor _tensor;
        int64_t _offset;
        int64_t _length;
    };

    struct HTDemucs::Priv
    {
       torch::Tensor _sources;
       ProgressUpdate fn = nullptr;
       void* progress_update_user = nullptr;
    };

    HTDemucs::HTDemucs(const char* model_path, const std::string& device, const std::string cache_dir)
        :
#ifdef ONNX_SUPPORT		
		_impl(std::make_shared<HTDemucs_impl>(model_path)), 
#endif
		_impl_ov(std::make_shared<HTDemucs_openvino_impl>(model_path, device, cache_dir)),
       _priv(std::make_shared<Priv>())
    {

    }

    static void dump_tensor(torch::Tensor z, const char* fname)
    {
        z = z.contiguous();
        std::ofstream wf(fname, std::ios::binary);
        wf.write((char*)z.data_ptr(), z.numel() * z.element_size());
        wf.close();
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

    static void spectro(torch::Tensor& x, torch::Tensor& z, int n_fft, int hop_length, int pad = 0)
    {
        std::vector<int64_t> other = { x.sizes()[0], x.sizes()[1] };
        int64_t length = x.sizes()[2];

        x = x.reshape({ -1, length });

        //std::cout << "x.shape after x.reshape({ -1, length }); = " << x.sizes() << std::endl;

        torch::Tensor hwindow = torch::hann_window(n_fft).to(x);

        //std::cout << "hwindow shape = " << hwindow.sizes() << std::endl;

       // at::Tensor stft(const at::Tensor & self, 
       //                  int64_t n_fft, 
       //                  c10::optional<int64_t> hop_length = c10::nullopt, 
       //                  c10::optional<int64_t> win_length = c10::nullopt, 
       //                  const c10::optional<at::Tensor> &window = {}, 
       //                  bool center = true, 
       //                  c10::string_view pad_mode = "reflect", 
       //                  bool normalized = false, 
       //                  c10::optional<bool> onesided = c10::nullopt,
       //                  c10::optional<bool> return_complex = c10::nullopt);
       //z = torch::stft(x, 
        //     n_fft * (1 + pad), hop_length, n_fft, hwindow, true, true, true, )
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

        //std::cout << "z.shape = " << z.sizes() << std::endl;

        int64_t freqs = z.sizes()[1];
        int64_t frame = z.sizes()[2];

        z = z.view({ other[0], other[1], freqs, frame });

        //std::cout << "z.shape after z.view = " << z.sizes() << std::endl;
    }

    static void ispectro(torch::Tensor& z, torch::Tensor& x, int64_t hop_length, int64_t length, int64_t pad = 0)
    {
        std::vector<int64_t> other = { z.sizes()[0], z.sizes()[1], z.sizes()[2] };
        int64_t freqs = z.sizes()[3];
        int64_t frames = z.sizes()[4];
        int64_t n_fft = 2 * freqs - 2;
        //std::cout << "z shape before z.view = " << z.sizes() << std::endl;
        z = z.view({ -1, freqs, frames });
        //std::cout << "z shape after z.view = " << z.sizes() << std::endl;



        int64_t win_length = n_fft / (1 + pad);

        //std::cout << "z.type = " << z.dtype() << std::endl;

        //std::cout << "z.to(torch::kFloat32).dtype = " << z.to(torch::kFloat32).dtype() << std::endl;
        //std::cout << "z.to(torch::kFloat32).sizes() = " << z.to(torch::kFloat32).sizes() << std::endl;

        //double check this..
        torch::Tensor hwindow = torch::hann_window(win_length).to(z.to(torch::kFloat32));
        //dump_tensor(hwindow, "my_hwindow.raw");

        //std::cout << "hwindow shape = " << hwindow.sizes() << std::endl;

        //std::cout << "torch::istft: " << std::endl;
        //std::cout << "    n_fft =  " << n_fft << std::endl;
        //std::cout << "    hop_length =  " << hop_length << std::endl;
        //std::cout << "    win_length =  " << win_length << std::endl;
        //std::cout << "    length =  " << length << std::endl;

        //std::cout << "z.dtype before istft = " << z.dtype() << std::endl;
        //dump_tensor(z, "my_complex_z.raw");

        x = torch::istft(z, n_fft, hop_length, win_length, hwindow, true, true, true, length);

        //static int dd = 0;
        //if (dd++ == 0)
        //{
        //    dump_tensor(x, "my_istft_out.raw");
        //}

        length = x.sizes()[1];
        x = x.view({ other[0], other[1], other[2], length });
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

        //std::cout << "x.shape before pad = " << x.sizes() << std::endl;

        pad1d(x, x, { pad, pad + le * hl - x.sizes()[2] });

        //std::cout << "x.shape after pad = " << x.sizes() << std::endl;

        //z = spectro(x, nfft, hl)[..., :-1, :]
        spectro(x, z, nfft, hl);

        z = z.index({ "...", torch::indexing::Slice(0, z.sizes()[2] - 1),  torch::indexing::Slice(torch::indexing::None) });

        //z = z[..., 2: 2 + le]
        z = z.index({ "...", torch::indexing::Slice(2, 2 + le) });

        //std::cout << "z.shape after slice = " << z.sizes() << std::endl;

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

        //std::cout << "le = " << le << std::endl;

        ispectro(z, x, hl, le);

        //std::cout << "ispec: return of ispectro shape = " << x.sizes() << std::endl;
        //dump_tensor(x, "my_ispectro_out.raw");

       // std::cout << "pad = " << pad << std::endl;
       // std::cout << "length = " << length << std::endl;
       // std::cout << "x.dtype  = " << x.dtype() << std::endl;

        x = x.index({ "...", torch::indexing::Slice(pad, pad + length) });

        //dump_tensor(x, "after_slice.raw");

        //std::cout << "ispec: x shape after slice = " << x.sizes() << std::endl;
    }

    bool HTDemucs::_actually_run_model(torch::Tensor& mix_tensor, torch::Tensor& x)
    {
        //length = mix.shape[-1]
        int64_t length = mix_tensor.sizes()[2];

        //segment = Fraction(39, 5)
        //samplerate = 44100
        //training_length = int(segment * samplerate)

        //samplerate = 44100
        int samplerate = 44100;
        int training_length = (44100 * 39) / 5; //343980

        torch::Tensor mix = mix_tensor;

        torch::Tensor z;
        spec(mix_tensor, z);

        torch::Tensor mag;
        magnitude(z, mag);

        //std::cout << "mag shape = " << mag.sizes() << std::endl;

        //dump_tensor(mag, "my_mag.raw");

        x = mag;

        int64_t B = x.sizes()[0];
        int64_t C = x.sizes()[1];
        int64_t Fq = x.sizes()[2];
        int64_t T = x.sizes()[3];

        //unlike previous Demucs, we always normalize because it is easier.
        torch::Tensor mean = x.mean({ 1, 2, 3 }, true);
        torch::Tensor std = x.std({ 1, 2, 3 }, true);
        x = (x - mean) / (1e-5 + std);
        //dump_tensor(x, "my_x_inference_input.raw");

        //std::cout << "x inference input shape = " << x.sizes() << std::endl;
        // x will be the freq.branch input.

        // Prepare the time branch input.
        torch::Tensor xt = mix;
        torch::Tensor meant = xt.mean({ 1, 2 }, true);
        torch::Tensor stdt = xt.std({ 1, 2 }, true);
        xt = (xt - meant) / (1e-5 + stdt);
        //dump_tensor(xt, "my_xt_inference_input.raw");

        //std::cout << "xt inference input shape = " << xt.sizes() << std::endl;

        //run inference here!
        //...
       //_impl->run_inference(x, xt);
       _impl_ov->run_inference(x, xt);

       _inference_i++;

       double perc_complete = ((double)_inference_i / (_shifts * _offsets)) * 100.0;

       if (_priv->fn)
       {
          if (!_priv->fn(perc_complete, _priv->progress_update_user))
          {
             return false;
          }
       }

       //static int dd = 0;
       //if (dd++ == 0)
       //{
       //    dump_tensor(x, "my_x_ov_out_tensor.raw");
       //    dump_tensor(xt, "my_xt_ov_out_tensor.raw");
       //}

        //post-process

        // std::cout << "x_out_tensor shape = " << x.sizes() << std::endl;
        // std::cout << "xt_out_tensor shape = " << xt.sizes() << std::endl;

        int64_t S = 4;
        x = x.view({ B, S, -1, Fq, T });

        //std::cout << "x_out_tensor shape (after x_out_tensor.view({ B, S, -1, training_length }) ) =  " << x.sizes() << std::endl;

        //x = x * std[:, None] + mean[:, None]
        x = x * std.index({ torch::indexing::Slice(), torch::indexing::None }) + mean.index({ torch::indexing::Slice(), torch::indexing::None });

        //dump_tensor(x_out_tensor, "my_x_out_after_slice.raw");

        torch::Tensor zout;
        mask_no_z(x, zout);

        //dump_tensor(zout, "my_zout.raw");

        ispec(zout, x, training_length);

        //std::cout << "here!" << std::endl;
        //std::cout << "x_out_tensor shape after ispec = " << x.sizes() << std::endl;
        //dump_tensor(x, "my_ispec_out.raw");

        xt = xt.view({ B, S, -1, training_length });
        xt = xt * stdt.index({ torch::indexing::Slice(), torch::indexing::None }) + meant.index({ torch::indexing::Slice(), torch::indexing::None });
        x = xt + x;

        //std::cout << "saving final output tensor, which has shape = " << x.sizes() << std::endl;
        //dump_tensor(x, "my_x_final_out.raw");
        return true;
    }

    static void center_trim(torch::Tensor& tensor, int64_t reference)
    {
        int64_t ref_size = reference;
        size_t dim_size = tensor.sizes().size();
        int64_t delta = tensor.sizes()[dim_size - 1] - ref_size;
        if (delta < 0)
        {
            std::cout << "uh oh! delta < 0!" << std::endl;
            return;
        }

        if (delta)
            tensor = tensor.index({ "...", torch::indexing::Slice(delta / 2, -(delta - delta / 2)) });
    }

    

    bool HTDemucs::_apply_model_3(TensorChunk& mix, torch::Tensor& out)
    {
        //std::cout << "apply_model_3->" << std::endl;

        int64_t batch = mix.shape()[0];
        int64_t channels = mix.shape()[1];
        int64_t length = mix.shape()[2];

        //valid_length = model.valid_length(length)
        int64_t valid_length = 343980;

        torch::Tensor padded_mix;
        mix.padded(valid_length, padded_mix);


        if (!_actually_run_model(padded_mix, out))
        {
           return false;
        }

        center_trim(out, length);

        return true;
    }

    bool HTDemucs::_apply_model_2(TensorChunk& mix, torch::Tensor& out, bool split, double overlap, double transition_power, int64_t static_shifts)
    {
        //std::cout << "apply_model_2->" << std::endl;

        int64_t batch = mix.shape()[0];
        int64_t channels = mix.shape()[1];
        int64_t length = mix.shape()[2];

        int64_t model_sources = 4;
        out = torch::zeros({ batch, 4, channels, length });

        torch::Tensor sum_weight = torch::zeros({ length });
        int64_t segment = (int64_t)((44100 * 39) / 5);
        int64_t stride = (int64_t)((1.0 - overlap) * segment);

        //std::cout << "offsets = " << std::endl;
        std::vector<int64_t> offsets;
        for (int64_t i = 0; i < length; i += stride)
        {
            offsets.push_back(i);
            //std::cout << i << std::endl;
        }

        //scale = float(format(stride / model.samplerate, ".2f"))
        double scale = round(((double)stride / 44100.0) * 100.0) / 100.0;

        // We start from a triangle shaped weight, with maximal weight in the middle
        // of the segment.Then we normalizeand take to the power `transition_power`.
        // Large values of transition power will lead to sharper transitions.
        torch::Tensor weight = torch::cat({ torch::arange(1, segment / 2 + 1),  torch::arange(segment - segment / 2, 0, -1) });
        //dump_tensor(weight, "my_weight.raw");

        // If the overlap < 50 %, this will translate to linear transition when
        // transition_power is 1.
        //    weight = (weight / weight.max()) * *transition_power
        weight = (weight / weight.max()).pow(transition_power);
        //dump_tensor(weight, "my_weight_after_pow.raw");

        _offsets = offsets.size();
        for (size_t i = 0; i < offsets.size(); i++)
        {
            TensorChunk chunk(mix, offsets[i], segment);

            torch::Tensor chunk_out;
            if (!_apply_model_3(chunk, chunk_out))
            {
               return false;
            }

            int64_t chunk_length = chunk_out.sizes().back();
            //out[..., offset:offset + segment] += (weight[:chunk_length] * chunk_out).to(mix.device)
            //dump_tensor(out, "out_before_plus.raw");
            out.index({ "...", torch::indexing::Slice(offsets[i], offsets[i] + segment) }) += weight.index({ torch::indexing::Slice(torch::indexing::None, chunk_length) }) * chunk_out;
            //dump_tensor(out, "out_after_plus.raw");
            sum_weight.index({ torch::indexing::Slice(offsets[i], offsets[i] + segment) }) += weight.index({ torch::indexing::Slice(torch::indexing::None, chunk_length) });

            //dump_tensor(sum_weight, "sum_weight.raw");

        }

        out /= sum_weight;

        // std::cout << "out.dtype() = " << out.dtype() << std::endl;


        //std::cout << "<-apply_model_2" << std::endl;
        return true;
    }

    bool HTDemucs::_apply_model_1(torch::Tensor& mix, torch::Tensor& out, int64_t shifts, bool split, double overlap, double transition_power, int64_t static_shifts)
    {
        //std::cout << "apply_model_1->" << std::endl;

        _shifts = shifts;

        int64_t length = mix.sizes().back();

        int64_t max_shift = 44100 / 2;
        TensorChunk mix_chunk(mix);

        torch::Tensor padded_mix;
        mix_chunk.padded(length + 2 * max_shift, padded_mix);

        //dump_tensor(padded_mix, "my_padded_mix.raw");

        for (int i = 0; i < shifts; i++)
        {
            //std::cout << "shift " << i << "->" << std::endl;
            //HACK
             int offset = rand() % max_shift;
            //int64_t offset = max_shift / 2;

            TensorChunk shifted(padded_mix, offset, length + max_shift - offset);
            torch::Tensor shifted_out;
            if (!_apply_model_2(shifted, shifted_out, split, overlap, transition_power, static_shifts))
            {
               return false;
            }

            if (i == 0)
                out = shifted_out.index({ "...", torch::indexing::Slice(max_shift - offset, torch::indexing::None) });
            else
                out += shifted_out.index({ "...", torch::indexing::Slice(max_shift - offset, torch::indexing::None) });
            //std::cout << "<-shift " << i << std::endl;
        }
        out /= shifts;

        //std::cout << "<-apply_model_1" << std::endl;
        return true;
    }

    bool HTDemucs::_apply_model_0(torch::Tensor& mix, torch::Tensor& out, int64_t shifts, bool split, double overlap, double transition_power, int64_t static_shifts)
    {
        _inference_i = 0;
        //std::cout << "apply_model_0->" << std::endl;
        return _apply_model_1(mix, out, shifts, split, overlap, transition_power, static_shifts);
        //std::cout << "<-apply_model_0" << std::endl;

        //For demucs4, we only apply 1 model and the weights are [1.0 1.0 1.0 1.0], so no need to implement
        // the 'for k, inst_weight in enumerate(weight):' stuff that we see in apply.py.
        // Once we get to implementing other variants of demuc4, it'll be necessary.

    }

   
    bool HTDemucs::Apply(float* pIn, int64_t nsamples,
                         float* &pOut0,
                         float* &pOut1,
                         float* &pOut2,
                         float* &pOut3,
                         int64_t num_shifts,
                         ProgressUpdate fn,
                         void* progress_update_user)
    {
         _priv->fn = fn;
         _priv->progress_update_user = progress_update_user;
         torch::Tensor cmix = torch::from_blob(pIn, { 2, nsamples });

         torch::Tensor ref = torch::mean(cmix, 0);
         //std::cout << "ref.sizes() = " << ref.sizes() << std::endl;
         //dump_tensor(ref, "my_ref.raw");
         cmix = (cmix - ref.mean()) / ref.std();
         torch::Tensor mix_infer = cmix.unsqueeze(0);
         //dump_tensor(mix_infer, "my_cmix_after_mean_std.raw");
         //std::cout << "mix_infer.sizes() = " << mix_infer.sizes() << std::endl;

         if (!_apply_model_0(mix_infer, _priv->_sources, num_shifts, true, 0.25, 1.0, 2))
         {
            return false;
         }
         _priv->_sources = _priv->_sources.squeeze(0);

         //dump_tensor(sources, "my_sources.raw");

         //sources = (sources * ref.std() + ref.mean()).cpu().numpy()
         _priv->_sources = _priv->_sources * ref.std() + ref.mean();

         //dump_tensor(sources, "my_sources_after_std_mean.raw");

         //std::cout << "sources.sizes() after _apply_model_0 = " << sources.sizes() << std::endl;

         _priv->_sources.contiguous();

         float* pSources = (float*)_priv->_sources.data_ptr();

         pOut0 = pSources;
         pSources += 2 * nsamples;
         pOut1 = pSources;
         pSources += 2 * nsamples;
         pOut2 = pSources;
         pSources += 2 * nsamples;
         pOut3 = pSources;

         return true;
    }

}
