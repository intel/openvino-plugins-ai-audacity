// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#include "dfnet_model.h"
#include "musicgen_utils.h"

namespace ov_deepfilternet
{
   static torch::Tensor erb_fb(torch::Tensor erb_widths, int64_t sr, bool normalized = true, bool inverse = false)
   {
      using namespace torch::indexing;

      auto n_freqs = torch::sum(erb_widths).item().toLong();;

      std::cout << "n_freqs = " << n_freqs << std::endl;

      auto all_freqs = torch::linspace(0, sr / 2, n_freqs + 1).index({ Slice(None, -1) });

      std::cout << "all_freqs shape = " << all_freqs.sizes() << std::endl;

      torch::Tensor zero = torch::zeros({ 1 }, torch::kInt64);
      torch::Tensor extended_widths = torch::cat({ zero, erb_widths });
      torch::Tensor cumsum = torch::cumsum(extended_widths, 0);
      torch::Tensor b_pts = cumsum.slice(0, 0, -1);

      std::cout << "b_pts shape = " << b_pts.sizes() << std::endl;

      torch::Tensor fb = torch::zeros({ all_freqs.size(0), b_pts.size(0) });
      for (int64_t i = 0; i < b_pts.size(0); ++i) {
         int64_t b = b_pts[i].item<int64_t>();
         int64_t w = erb_widths[i].item<int64_t>();

         fb.index({ torch::indexing::Slice(b, b + w), i }) = 1;
      }

      // Normalize to constant energy per resulting band
      if (inverse)
      {
         fb = fb.t();
         if (!normalized)
         {
            fb /= fb.sum(1, true);
         }
      }
      else
      {
         if (normalized)
         {
            fb /= fb.sum(0);
         }
      }

      return fb;
   }

   class Mask
   {
   public:

      Mask(torch::Tensor erb_inv_fb, bool post_filter = false, double eps = 1e-12)
      {
         _erb_inv_fb = erb_inv_fb;
         _post_filter = post_filter;
         _eps = eps;
      }

      torch::Tensor forward(torch::Tensor spec, torch::Tensor mask, std::optional<torch::Tensor> atten_lim = {})
      {
         if (_post_filter)
         {
            throw std::runtime_error("post_filter not implemented!");
         }

         if (atten_lim)
         {
            throw std::runtime_error("atten_lim path not implemented!");
         }

         mask = mask.matmul(_erb_inv_fb);
         if (!spec.is_complex())
         {
            mask = mask.unsqueeze(4);
         }

         return spec * mask;
      }
   private:

      torch::Tensor _erb_inv_fb;
      bool _post_filter;
      double _eps;

   };

   DFNetModel::DFNetModel(std::string model_folder, std::string device, ModelSelection model_selection,
      std::optional<std::string> openvino_cache_dir, torch::Tensor erb_widths, int64_t lookahead, int64_t nb_df)
      : _nb_df(nb_df), _df(nb_df, 5, 2)
   {
      auto erb_inv_fb = erb_fb(erb_widths, 48000, true, true);

      _bDF3 = (model_selection == ModelSelection::DEEPFILTERNET3);

      if (_bDF3)
      {
         model_folder = FullPath(model_folder, "deepfilternet3");
      }
      else
      {
         model_folder = FullPath(model_folder, "deepfilternet2");
      }

      if (lookahead > 0)
      {
         _pad_feat = std::make_shared<torch::nn::ConstantPad2d>(torch::nn::ConstantPad2dOptions({ 0, 0, -lookahead, lookahead }, 0.0));
      }

      _mask = std::make_shared<Mask>(erb_inv_fb);

      _core = std::make_shared< ov::Core >();

      if (openvino_cache_dir)
      {
         _core->set_property(ov::cache_dir(*openvino_cache_dir));
      }

      _num_hops = 3002;

#define DFNET_USE_ONNX 0 //set to 1 to use .onnx models, set to 0 to use OpenVINO IR (.xml)

      //enc
      {
#if DFNET_USE_ONNX
         auto model_fullpath = FullPath(model_folder, "enc.onnx");
#else
         auto model_fullpath = FullPath(model_folder, "enc.xml");
#endif
         auto model = _core->read_model(model_fullpath);

         std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
         port_to_shape[model->input("feat_erb")] = { 1, 1, _num_hops, 32 };
         port_to_shape[model->input("feat_spec")] = { 1, 2, _num_hops, 96 };
         model->reshape(port_to_shape);

         std::cout << "enc: " << std::endl;
         logBasicModelInfo(model);
#if 0
         //dump IR
         ov::serialize(model, "enc.xml", "enc.bin");
#endif
         auto compiledModel = _core->compile_model(model, device);
         _infer_request_enc = compiledModel.create_infer_request();
      }

      //erb_dec
      {
#if DFNET_USE_ONNX
         auto model_fullpath = FullPath(model_folder, "erb_dec.onnx");
#else
         auto model_fullpath = FullPath(model_folder, "erb_dec.xml");
#endif
         auto model = _core->read_model(model_fullpath);

         std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
         if (_bDF3)
         {
            port_to_shape[model->input("emb")] = { 1, _num_hops, 512 };
         }
         else
         {
            port_to_shape[model->input("emb")] = { 1, _num_hops, 256 };
         }

         port_to_shape[model->input("e3")] = { 1, 64, _num_hops, 8 };
         port_to_shape[model->input("e2")] = { 1, 64, _num_hops, 8 };
         port_to_shape[model->input("e1")] = { 1, 64, _num_hops, 16 };
         port_to_shape[model->input("e0")] = { 1, 64, _num_hops, 32 };
         model->reshape(port_to_shape);
#if 0
         //dump IR
         ov::serialize(model, "erb_dec.xml", "erb_dec.bin");
#endif
         std::cout << "erb_dec: " << std::endl;
         logBasicModelInfo(model);
         auto compiledModel = _core->compile_model(model, device);
         _infer_request_erb_dec = compiledModel.create_infer_request();

         //'link' the output of enc directly to the input of erb_dec
         _infer_request_erb_dec.set_tensor("emb", _infer_request_enc.get_tensor("emb"));
         _infer_request_erb_dec.set_tensor("e3", _infer_request_enc.get_tensor("e3"));
         _infer_request_erb_dec.set_tensor("e2", _infer_request_enc.get_tensor("e2"));
         _infer_request_erb_dec.set_tensor("e1", _infer_request_enc.get_tensor("e1"));
         _infer_request_erb_dec.set_tensor("e0", _infer_request_enc.get_tensor("e0"));
      }

      //df_dec
      {
#if DFNET_USE_ONNX
         auto model_fullpath = FullPath(model_folder, "df_dec.onnx");
#else
         auto model_fullpath = FullPath(model_folder, "df_dec.xml");
#endif
         auto model = _core->read_model(model_fullpath);

         std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
         if (_bDF3)
         {
            port_to_shape[model->input("emb")] = { 1, _num_hops, 512 };
         }
         else
         {
            port_to_shape[model->input("emb")] = { 1, _num_hops, 256 };
         }
         port_to_shape[model->input("c0")] = { 1, 64, _num_hops, 96 };
         model->reshape(port_to_shape);
#if 0
         //dump IR
         ov::serialize(model, "df_dec.xml", "df_dec.bin");
#endif
         std::cout << "df_dec: " << std::endl;
         logBasicModelInfo(model);
         auto compiledModel = _core->compile_model(model, device);
         _infer_request_df_dec = compiledModel.create_infer_request();

         _infer_request_df_dec.set_tensor("emb", _infer_request_enc.get_tensor("emb"));
         _infer_request_df_dec.set_tensor("c0", _infer_request_enc.get_tensor("c0"));
      }
   }

   torch::Tensor DFNetModel::forward(torch::Tensor spec, torch::Tensor feat_erb, torch::Tensor feat_spec, bool post_filter)
   {

      if (_bDF3)
      {
         return forward_df3(spec, feat_erb, feat_spec, post_filter);
      }
      else
      {
         return forward_df2(spec, feat_erb, feat_spec);
      }
   }

   static inline torch::Tensor as_complex(torch::Tensor x)
   {
      if (torch::is_complex(x))
         return x;

      if (x.size(-1) != 2)
      {
         throw std::runtime_error("Last dimension need to be of length 2 (re + im)");
      }

      if (x.stride(-1) != 1)
      {
         x = x.contiguous();
      }

      return torch::view_as_complex(x);
   }

   torch::Tensor DFNetModel::forward_df3(torch::Tensor spec, torch::Tensor feat_erb, torch::Tensor feat_spec, bool post_filter)
   {

      feat_spec = feat_spec.squeeze(1).permute({ 0, 3, 1, 2 });
      feat_erb = (*_pad_feat)(feat_erb);
      feat_spec = (*_pad_feat)(feat_spec);

      //run enc
      {
         auto ov_erb = wrap_ov_tensor_as_torch(_infer_request_enc.get_tensor("feat_erb"));
         auto ov_feat_spec = wrap_ov_tensor_as_torch(_infer_request_enc.get_tensor("feat_spec"));

         ov_erb.copy_(feat_erb);
         ov_feat_spec.copy_(feat_spec);

         _infer_request_enc.infer();

         auto lsnr = wrap_ov_tensor_as_torch(_infer_request_enc.get_tensor("lsnr"));

         //note: remember, the output tensors of _infer_request_enc are set as input tensors for 
         // both _infer_request_erb_dec and _infer_request_df_dec, which is why you don't see me explictly
         // grabbing the output of the above infer, and copying them to the input tensors for the other 
         // infer_requests.
      }

      //expose this?
      bool run_erb = true;

      torch::Tensor m, spec_m;
      if (run_erb)
      {
         _infer_request_erb_dec.infer();
         m = wrap_ov_tensor_as_torch(_infer_request_erb_dec.get_tensor("m"));

         //auto pad_spec = torch::nn::functional::pad(spec, torch::nn::functional::PadFuncOptions({ 0, 0, 0, 0, 1, -1, 0, 0 }).value(0.0));

         spec_m = _mask->forward(spec, m);

      }
      else
      {
         //todo?
         throw std::runtime_error("not implemented run_erb=false");
      }

      //expose this?
      bool run_df = true;

      torch::Tensor df_coefs;
      if (run_df)
      {
         _infer_request_df_dec.infer();

         df_coefs = wrap_ov_tensor_as_torch(_infer_request_df_dec.get_tensor("coefs"));

         //DfOutputReshapeMF forward
         {

            std::vector< int64_t > new_shape;
            for (size_t i = 0; i < df_coefs.sizes().size(); i++)
            {
               new_shape.push_back(df_coefs.size(i));
            }

            new_shape[new_shape.size() - 1] = -1;
            new_shape.push_back(2);

            df_coefs = df_coefs.view(new_shape);
            df_coefs = df_coefs.permute({ 0, 3, 1, 2, 4 });
         }
      }
      else
      {
         throw std::runtime_error("not implemented run_df=false");
      }

      using namespace torch::indexing;
      auto spec_e = _df.forward(spec.clone(), df_coefs);
      spec_e.index_put_({ "...", Slice(_nb_df, None),  Slice(None) }, spec_m.index({ "...", Slice(_nb_df, None),  Slice(None) }));

      if (post_filter)
      {
         float beta = 0.02f;
         float eps = 1e-12f;
         auto mask = (as_complex(spec_e).abs() / as_complex(spec).abs().add(eps)).clamp(eps, 1);
         auto mask_sin = mask * torch::sin(M_PI * mask / 2).clamp_min(eps);
         auto pf = (1 + beta) / (1 + beta * mask.div(mask_sin).pow(2));
         spec_e = spec_e * pf.unsqueeze(-1);
      }

      return spec_e;
   }

   torch::Tensor DFNetModel::forward_df2(torch::Tensor spec, torch::Tensor feat_erb, torch::Tensor feat_spec)
   {
      feat_spec = feat_spec.squeeze(1).permute({ 0, 3, 1, 2 });
      feat_erb = (*_pad_feat)(feat_erb);
      feat_spec = (*_pad_feat)(feat_spec);

      //run enc
      {
         auto ov_erb = wrap_ov_tensor_as_torch(_infer_request_enc.get_tensor("feat_erb"));
         auto ov_feat_spec = wrap_ov_tensor_as_torch(_infer_request_enc.get_tensor("feat_spec"));

         ov_erb.copy_(feat_erb);
         ov_feat_spec.copy_(feat_spec);

         _infer_request_enc.infer();


         //note: remember, the output tensors of _infer_request_enc are set as input tensors for 
         // both _infer_request_erb_dec and _infer_request_df_dec, which is why you don't see me explictly
         // grabbing the output of the above infer, and copying them to the input tensors for the other 
         // infer_requests.
      }

      //expose this?
      bool run_erb = true;
      {
         _infer_request_erb_dec.infer();
         auto m = wrap_ov_tensor_as_torch(_infer_request_erb_dec.get_tensor("m"));

         //auto pad_spec = torch::nn::functional::pad(spec, torch::nn::functional::PadFuncOptions({ 0, 0, 0, 0, 1, -1, 0, 0 }).value(0.0));

         spec = _mask->forward(spec, m);
      }

      //expose this?
      bool run_df = true;

      torch::Tensor df_coefs;
      if (run_df)
      {
         _infer_request_df_dec.infer();

         df_coefs = wrap_ov_tensor_as_torch(_infer_request_df_dec.get_tensor("coefs"));

         //DfOutputReshapeMF forward
         {

            std::vector< int64_t > new_shape;
            for (size_t i = 0; i < df_coefs.sizes().size(); i++)
            {
               new_shape.push_back(df_coefs.size(i));
            }

            new_shape[new_shape.size() - 1] = -1;
            new_shape.push_back(2);

            df_coefs = df_coefs.view(new_shape);
            df_coefs = df_coefs.permute({ 0, 3, 1, 2, 4 });
         }
      }
      else
      {
         throw std::runtime_error("not implemented run_df=false");
      }


      using namespace torch::indexing;
      auto spec_e = _df.forward(spec, df_coefs);

      return spec_e;
   }


}
