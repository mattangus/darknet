#pragma once

#include <torch/torch.h>
#include "params/layers.hpp"
#include "model/pytorch/dark_module.hpp"

namespace darknet
{
namespace model
{
namespace pytorch
{
    class MaxPool : public DarknetModule
    {
    private:
        /* data */
        torch::nn::MaxPool2d pool{nullptr};

        int _stride, _kernel_size;

        torch::Tensor _forward(torch::Tensor x) {	
            if (_stride != 1)
            {
                x = torch::max_pool2d(x, {_kernel_size, _kernel_size}, {_stride, _stride});
            }
            else
            {
                int pad = _kernel_size - 1;

                torch::Tensor padded_x = torch::replication_pad2d(x, {0, pad, 0, pad});
                x = torch::max_pool2d(padded_x, {_kernel_size, _kernel_size}, {_stride, _stride});
            }       

            return x;
        }
    public:
        MaxPool(std::shared_ptr<params::MaxPoolParams>& params, std::vector<int>& outputDepths) : DarknetModule("MaxPool") {
            auto opt = torch::nn::MaxPool2dOptions(params->size);
            std::vector<int64_t> v = {params->stride_x, params->stride_y};
            opt.stride(v);
            opt.padding(params->padding);
            opt.dilation(1);

            _stride = params->stride_x;
            _kernel_size = params->size;

            pool = register_module("pool", torch::nn::MaxPool2d(opt));

            outputDepths.push_back(outputDepths.back());
        }
        ~MaxPool() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            return _forward(outputs.back());
        }

        void loadWeights(std::shared_ptr<weights::BinaryReader>& weightsReader) override
        {
            return;
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
