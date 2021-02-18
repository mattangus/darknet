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
        int layer_num;
    public:
        Route(std::shared_ptr<params::RouteParams>& params, std::vector<int>& outputDepths) : DarknetModule("Route") {
            assert(params->groups == 1); // only support groups of 1 for now
            layers = params->layers;
            layer_num = params->layer_num;
            
            int filterSum = 0;
            for(int i = 0; i < layers.size(); i++)
                filterSum += outputDepths[layers[i] + 1];
            outputDepths.push_back(filterSum);
        }
        ~Route() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            std::vector<torch::Tensor> tensors;
            tensors.reserve(layers.size());
            for(int i = 0; i < layers.size(); i++)
            {
                auto temp = outputs[layers[i] + 1]; // input layer is in the outputs list
                tensors.push_back(temp);
            }
            return torch::cat(tensors, 1);
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
