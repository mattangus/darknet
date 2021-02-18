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
    public:
        MaxPool(std::shared_ptr<params::MaxPoolParams>& params, std::vector<int>& outputDepths) : DarknetModule("MaxPool") {
            auto opt = torch::nn::MaxPool2dOptions(params->size);
            std::vector<int64_t> v = {params->stride_x, params->stride_y};
            opt.stride(v);
            opt.padding(params->padding);
            opt.dilation(1);

            pool = register_module("pool", torch::nn::MaxPool2d(opt));

            outputDepths.push_back(outputDepths.back());
        }
        ~MaxPool() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            return pool->forward(outputs.back());
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
