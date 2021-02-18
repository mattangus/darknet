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
    class Route : public DarknetModule
    {
    private:
        /* data */
        std::vector<int> layers;
    public:
        Route(std::shared_ptr<params::RouteParams>& params) {
            assert(params->groups == 1); // only support groups of 1 for now
            layers = params->layers;

        }
        ~Route() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            std::vector<torch::Tensor> tensors(layers.size());
            for(int i = 0; i < layers.size(); i++)
                tensors[i] = outputs[layers[i]];
            return torch::cat(tensors, -1);
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
