#include "static_kv_cache_manager_cl.h"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"

namespace ov_musicgen
{
   class StaticKVCacheManagerCL::Impl
   {
   public:

      Impl(ov::intel_gpu::ocl::ClContext gctx)
         : gpu_context(gctx)
      {
         context = gpu_context.get();
         cl::Device cl_device = cl::Device(context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);
         cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
         queue = cl::CommandQueue(context, cl_device, props);
      }

      ov::intel_gpu::ocl::ClContext gpu_context;
      cl::Context context;
      cl::CommandQueue queue;
      std::vector< std::vector< cl::Buffer > > _past_key_values_cl;
      std::vector< std::vector< cl::Buffer > > _new_key_values_cl;
      std::vector< std::vector< cl::Buffer > > _new_key_values_large_cl;

      cl::Buffer _make_tensor_remote_blob(ov::InferRequest infer_request, std::string tensorname)
      {
         auto ov_tensor = infer_request.get_tensor(tensorname);
         auto tensor_byte_size = ov_tensor.get_byte_size();
         cl_int err;
         cl::Buffer cl_buf(context, CL_MEM_READ_WRITE, tensor_byte_size, NULL, &err);
         auto shared_blob = gpu_context.create_tensor(ov_tensor.get_element_type(), ov_tensor.get_shape(), cl_buf);
         infer_request.set_tensor(tensorname, shared_blob);
         return cl_buf;
      }

   };

   StaticKVCacheManagerCL::StaticKVCacheManagerCL(ov::Core& core,
      ov::InferRequest infer_request_initial,
      ov::InferRequest infer_request_with_past,
      ov::InferRequest infer_request_without_past,
      const MusicgenDecoder::Config& decoder_config)
      : StaticKVCacheManager(infer_request_initial, infer_request_with_past,
         infer_request_without_past, decoder_config)
   {

      _impl = std::make_shared< Impl >(core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>());
   }

   void StaticKVCacheManagerCL::Init()
   {
      std::cout << "StaticKVCacheManagerCL::Init!" << std::endl;
      //allocate an OpenCL buffer for each of the input past_key_value tensors
      _impl->_past_key_values_cl.resize(_decoder_config.num_hidden_layers);
      _impl->_new_key_values_cl.resize(_decoder_config.num_hidden_layers);
      _impl->_new_key_values_large_cl.resize(_decoder_config.num_hidden_layers);

      for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
      {
         _impl->_past_key_values_cl[layeri].resize(4);

         {
            std::string past_base_name = "past_key_values." + std::to_string(layeri) + ".decoder.";
            std::string past_key_name = past_base_name + "key";
            std::string past_value_name = past_base_name + "value";

            {
               auto cl_buf = _impl->_make_tensor_remote_blob(_infer_request, past_key_name);
               _impl->_past_key_values_cl[layeri][0] = cl_buf;
            }

            {
               auto cl_buf = _impl->_make_tensor_remote_blob(_infer_request, past_value_name);
               _impl->_past_key_values_cl[layeri][1] = cl_buf;
            }

         }

         {
            std::string past_base_name = "past_key_values." + std::to_string(layeri) + ".encoder.";
            std::string past_key_name = past_base_name + "key";
            std::string past_value_name = past_base_name + "value";

            {
               auto cl_buf = _impl->_make_tensor_remote_blob(_infer_request, past_key_name);
               _impl->_past_key_values_cl[layeri][2] = cl_buf;
            }

            {
               auto cl_buf = _impl->_make_tensor_remote_blob(_infer_request, past_value_name);
               _impl->_past_key_values_cl[layeri][3] = cl_buf;
            } 
         }

         _impl->_new_key_values_cl[layeri].resize(2);
         _impl->_new_key_values_large_cl[layeri].resize(2);

         {
            std::string present_base_name = "present." + std::to_string(layeri) + ".decoder.";
            std::string present_key_name = present_base_name + "key";
            std::string present_value_name = present_base_name + "value";

            {
               auto cl_buf = _impl->_make_tensor_remote_blob(_infer_request, present_key_name);
               _impl->_new_key_values_cl[layeri][0] = cl_buf;
            }

            {
               auto cl_buf = _impl->_make_tensor_remote_blob(_infer_request, present_value_name);
               _impl->_new_key_values_cl[layeri][1] = cl_buf;
            }

            {
               auto cl_buf = _impl->_make_tensor_remote_blob(_infer_request_nonkv, present_key_name);
               _impl->_new_key_values_large_cl[layeri][0] = cl_buf;
            }

            {
               auto cl_buf = _impl->_make_tensor_remote_blob(_infer_request_nonkv, present_value_name);
               _impl->_new_key_values_large_cl[layeri][1] = cl_buf;
            }
         }
      }

      StaticKVCacheManager::Init();
   }

   void StaticKVCacheManagerCL::Reset()
   {
      for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
      {
         for (size_t i = 0; i < 4; i++)
         {
            int pattern = 0;
            auto buf = _impl->_past_key_values_cl[layeri][i];
            size_t size = 0;
            buf.getInfo(CL_MEM_SIZE, &size);
            _impl->queue.enqueueFillBuffer(buf, pattern, 0, size);
         }
      }
      _impl->queue.finish();
   }

   void StaticKVCacheManagerCL::UpdateFromSingle(size_t position)
   {
      auto past_key_values_shape = past_decoder_keys[0].get_shape();
      for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
      {
         for (int i = 0; i < 2; i++)
         {
            //slice the new key values into the existing past_key_vals buffer using OpenCL.
            std::array<size_t, 3> srcOrigin = { 0, 0, 0 }; // Start at the beginning of the source buffer
            std::array<size_t, 3> dstOrigin = { 0, position,  0 };

            // Size of one element
            std::array<size_t, 3> region = { sizeof(ov::float16) * past_key_values_shape[3], 1, past_key_values_shape[0] * past_key_values_shape[1] };

            size_t srcRowPitch = 64 * sizeof(ov::float16); // Size of one row in the source buffer
            size_t srcSlicePitch = 64 * sizeof(ov::float16) * 1; // Size of one 2D plane in the source buffer
            size_t dstRowPitch = 64 * sizeof(ov::float16); // Size of one row in the destination buffer
            size_t dstSlicePitch = 64 * sizeof(ov::float16) * past_key_values_shape[2]; // Size of one 2D plane in the destination buffer

            auto new_key_values = _impl->_new_key_values_cl[layeri][i];
            auto past_key_values = _impl->_past_key_values_cl[layeri][i];

            cl_int ret = _impl->queue.enqueueCopyBufferRect(new_key_values, past_key_values, srcOrigin, dstOrigin, region, srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch);

         }
      }

      _impl->queue.finish();
   }

   void StaticKVCacheManagerCL::UpdateFromLargeContext()
   {
      //insert the new key / value tensors into the past keys tensor
      auto past_key_values_shape = past_decoder_keys[0].get_shape();
      auto new_key_values_shape = present_decoder_keys_large_context[0].get_shape();
      for (size_t layeri = 0; layeri < _decoder_config.num_hidden_layers; layeri++)
      {
         for (int i = 0; i < 2; i++)
         {
            //slice the new key values into the past_key_vals buffer using OpenCL.
            std::array<size_t, 3> srcOrigin = { 0, 0, 0 }; // Start at the beginning of the source buffer
            std::array<size_t, 3> dstOrigin = { 0, 0, 0 }; // Start at the beginning of the destination buffer

            // Size of one element
            std::array<size_t, 3> region = { sizeof(ov::float16) * past_key_values_shape[3], new_key_values_shape[2], past_key_values_shape[0] * past_key_values_shape[1] };

            size_t srcRowPitch = past_key_values_shape[3] * sizeof(ov::float16); // Size of one row in the source buffer
            size_t srcSlicePitch = srcRowPitch * new_key_values_shape[2]; // Size of one 2D plane in the source buffer
            size_t dstRowPitch = past_key_values_shape[3] * sizeof(ov::float16); // Size of one row in the destination buffer
            size_t dstSlicePitch = dstRowPitch * past_key_values_shape[2]; // Size of one 2D plane in the destination buffer

            auto new_key_values = _impl->_new_key_values_large_cl[layeri][i];
            auto past_key_values = _impl->_past_key_values_cl[layeri][i];

            cl_int ret = _impl->queue.enqueueCopyBufferRect(new_key_values, past_key_values, srcOrigin, dstOrigin, region, srcRowPitch, srcSlicePitch, dstRowPitch, dstSlicePitch);
         }
      }

      _impl->queue.finish();
   }

}
