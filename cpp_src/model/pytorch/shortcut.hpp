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
        Shortcut(std::shared_ptr<params::ShortcutParams>& params) {
            layers = params->layers;
        }
        ~Shortcut() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            torch::Tensor output = outputs[layers[0]];
            for(int i = 1; i < layers.size(); i++)
                output += outputs[layers[i]];
            return output;
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
