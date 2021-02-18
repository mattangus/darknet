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
    class Upsample : public DarknetModule
    {
    private:
        torch::nn::Upsample up{nullptr};

    public:
        Upsample(std::shared_ptr<params::UpsampleParams>& params) {
            auto opt = torch::nn::UpsampleOptions();
            std::vector<int64_t> v = {params->stride, params->stride, 1};
            opt.size(v);
            
            up = register_module("upsample", torch::nn::Upsample(opt));
        }
        ~Upsample() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            return up->forward(outputs.back());
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
