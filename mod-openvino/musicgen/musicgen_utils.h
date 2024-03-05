// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <iostream>
#include <openvino/openvino.hpp>
#include <chrono>
#include <optional>
#include <torch/torch.h>
#include <fstream>
#include <codecvt>
#include <locale>

static void dump_tensor(torch::Tensor z, std::string fname)
{
   z = z.contiguous();
   std::ofstream wf(fname, std::ios::binary);
   wf.write((char*)z.data_ptr(), z.numel() * z.element_size());
   wf.close();
}

static inline void save_tensor_to_disk(ov::Tensor& t, std::string filename)
{
   std::ofstream wf(filename.c_str(), std::ios::out | std::ios::binary);
   if (!wf)
   {
      std::cout << "could not open file for writing" << std::endl;
      return;
   }

   size_t total_bytes = t.get_byte_size();
   void* pTData = t.data();
   wf.write((char*)pTData, total_bytes);
   wf.close();
}

#ifdef WIN32
#define OS_SEP "\\"
#else
#define OS_SEP "/"
#endif

static inline std::string FullPath(std::string base_dir, std::string filename)
{
   return base_dir + OS_SEP + filename;
}

static inline std::wstring ToWString(const std::string& str)
{
   return std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(str);
}

static ov::Tensor wrap_torch_tensor_as_ov(torch::Tensor tensor_torch)
{

   if (!tensor_torch.defined())
   {
      throw std::runtime_error("wrap_torch_tensor_as_ov: received undefined tensor");
   }
   size_t element_byte_size;
   void* pData = tensor_torch.data_ptr();
   ov::element::Type ov_element_type;
   switch (tensor_torch.dtype().toScalarType())
   {
   case torch::kFloat32:
      ov_element_type = ov::element::f32;
      element_byte_size = sizeof(float);
      break;

   case torch::kFloat16:
      ov_element_type = ov::element::f16;
      element_byte_size = sizeof(short);
      break;

   case torch::kInt64:
      ov_element_type = ov::element::i64;
      element_byte_size = sizeof(int64_t);
      break;
   default:
      std::cout << "type = " << tensor_torch.dtype() << std::endl;
      throw std::invalid_argument("wrap_torch_tensor_as_ov: unsupported type");
      break;
   }

   std::vector<size_t> ov_shape;
   for (auto s : tensor_torch.sizes())
      ov_shape.push_back(s);

   //OV strides are in bytes, whereas torch strides are in # of elements.
   std::vector<size_t> ov_strides;
   for (auto s : tensor_torch.strides())
      ov_strides.push_back(s * element_byte_size);

   return ov::Tensor(ov_element_type, ov_shape, pData, ov_strides);
}

static torch::Tensor wrap_ov_tensor_as_torch(ov::Tensor ov_tensor)
{
   if (!ov_tensor)
   {
      throw std::invalid_argument("wrap_ov_tensor_as_torch: invalid ov_tensor");
   }

   //first, determine torch dtype from ov type
   at::ScalarType torch_dtype;
   size_t element_byte_size;
   void* pOV_Tensor;
   switch (ov_tensor.get_element_type())
   {
   case ov::element::i8:
      torch_dtype = torch::kI8;
      element_byte_size = sizeof(unsigned char);
      pOV_Tensor = ov_tensor.data();
      break;

   case ov::element::f32:
      torch_dtype = torch::kFloat32;
      element_byte_size = sizeof(float);
      pOV_Tensor = ov_tensor.data<float>();
      break;

   case ov::element::f16:
      torch_dtype = torch::kFloat16;
      element_byte_size = sizeof(short);
      pOV_Tensor = ov_tensor.data<ov::float16>();
      break;

   case ov::element::i64:
      torch_dtype = torch::kInt64;
      element_byte_size = sizeof(int64_t);
      pOV_Tensor = ov_tensor.data<int64_t>();
      break;

   default:
      std::cout << "type = " << ov_tensor.get_element_type() << std::endl;
      throw std::invalid_argument("wrap_ov_tensor_as_torch: unsupported type");
      break;
   }

   //fill torch shape
   std::vector<int64_t> torch_shape;
   for (auto s : ov_tensor.get_shape())
      torch_shape.push_back(s);

   std::vector<int64_t> torch_strides;
   for (auto s : ov_tensor.get_strides())
      torch_strides.push_back(s / element_byte_size); //<- torch stride is in term of # of elements, whereas openvino stride is in terms of bytes

   auto options =
      torch::TensorOptions()
      .dtype(torch_dtype);

   return torch::from_blob(pOV_Tensor, torch_shape, torch_strides, options);
}

static torch::Tensor read_tensor(std::string filename, at::IntArrayRef shape, at::ScalarType dtype = torch::kFloat32)
{
   uint64_t nelements = 1;
   for (auto s : shape)
   {
      nelements *= s;
   }

   size_t size_of_element;
   switch (dtype)
   {
   case torch::kFloat32: size_of_element = sizeof(float); break;
   case torch::kInt64: size_of_element = sizeof(int64_t); break;
   default:
      throw std::invalid_argument("read_tensor: unsupported dtype");
      break;

   }

   void* pTensorData = malloc(nelements * sizeof(size_of_element));

#ifdef WIN32
   std::ifstream rf(ToWString(filename), std::ios::binary);
#else
   std::ifstream rf(filename, std::ios::binary);
#endif
   if (!rf.is_open())
   {
      throw std::runtime_error("Unable to open: " + filename);
   }
   rf.read((char*)pTensorData, nelements * size_of_element);
   rf.close();

   auto options =
      torch::TensorOptions()
      .dtype(dtype);

   return torch::from_blob(pTensorData, shape, options);
}

static inline void print_compiled_model_properties(ov::CompiledModel& compiledModel)
{
   auto supported_properties = compiledModel.get_property(ov::supported_properties);
   std::cout << "Model:" << std::endl;
   for (const auto& cfg : supported_properties) {
      if (cfg == ov::supported_properties)
         continue;
      auto prop = compiledModel.get_property(cfg);
      if (cfg == ov::device::properties) {
         auto devices_properties = prop.as<ov::AnyMap>();
         for (auto& item : devices_properties) {
            std::cout << "  " << item.first << ": " << std::endl;
            for (auto& item2 : item.second.as<ov::AnyMap>()) {
               OPENVINO_SUPPRESS_DEPRECATED_START
                  if (item2.first == ov::supported_properties || item2.first == "SUPPORTED_CONFIG_KEYS)" ||
                     item2.first == "SUPPORTED_METRICS")
                     continue;
               OPENVINO_SUPPRESS_DEPRECATED_END
                  std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
            }
         }
      }
      else {
         std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
      }
   }
}

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

static ov::hint::PerformanceMode get_performance_hint(const std::string& device, const std::string& hint, const ov::Core& core)
{
   ov::hint::PerformanceMode ov_perf_hint = ov::hint::PerformanceMode::LATENCY;

   auto supported_properties = core.get_property(device, ov::supported_properties);
   if (std::find(supported_properties.begin(), supported_properties.end(), ov::hint::performance_mode) !=
      supported_properties.end()) {
      if (hint != "") {
         if (hint == "throughput" || hint == "tput") {
            ov_perf_hint = ov::hint::PerformanceMode::THROUGHPUT;
         }
         else if (hint == "latency") {
            ov_perf_hint = ov::hint::PerformanceMode::LATENCY;
         }
         else if (hint == "cumulative_throughput" || hint == "ctput") {
            ov_perf_hint = ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
         }
         else {
            throw std::runtime_error(
               "Incorrect performance hint string. Supported string are "
               "`throughput`(tput), `latency', 'cumulative_throughput'(ctput) value or 'none'.");
         }
      }
   }
   else
   {
      throw std::runtime_error("could not find performance mode in supported properties for device = " + device);
   }

   return ov_perf_hint;
}
