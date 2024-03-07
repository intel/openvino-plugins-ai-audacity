#pragma once

#include <torch/torch.h>

namespace ov_deepfilternet
{
	class DFNet
	{
	public:

		virtual torch::Tensor
			forward(torch::Tensor spec, torch::Tensor feat_erb, torch::Tensor feat_spec) = 0;
	
	
		virtual int64_t num_static_hops() = 0;
	
	};
}