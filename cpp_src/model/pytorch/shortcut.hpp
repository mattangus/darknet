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
    class Shortcut : public DarknetModule
    {
    private:
        /* data */
        std::vector<int> layers;
    public:
        Shortcut(std::shared_ptr<params::ShortcutParams>& params, std::vector<int>& outputDepths) : DarknetModule("Shortcut") {
            layers = params->layers;

            int depth = 0;
            for(int i = 0; i < layers.size(); i++)
                depth += outputDepths[layers[i] + 1]; // input layer is in the outputs list
            outputDepths.push_back(depth);
        }
        ~Shortcut() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            torch::Tensor output = outputs[layers[0] + 1];
            for(int i = 1; i < layers.size(); i++)
                output += outputs[layers[i] + 1]; // input layer is in the outputs list
            return output;
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
