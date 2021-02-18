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
        MaxPool(std::shared_ptr<params::MaxPoolParams>& params) {
            auto opt = torch::nn::MaxPool2dOptions(params->size);
            opt.stride(params->stride);

            pool = register_module("pool", torch::nn::MaxPool2d(opt));
        }
        ~MaxPool() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            return pool->forward(outputs.back());
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
