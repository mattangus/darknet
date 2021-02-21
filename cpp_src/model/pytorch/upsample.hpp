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
        Upsample(std::shared_ptr<params::UpsampleParams>& params, std::vector<int>& outputDepths) : DarknetModule("Upsample") {
            auto opt = torch::nn::UpsampleOptions();
            std::vector<double> v = {(double)params->stride, (double)params->stride};
            opt.scale_factor(v);
            
            up = register_module("upsample", torch::nn::Upsample(opt));

            outputDepths.push_back(outputDepths.back());
        }
        ~Upsample() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            return up->forward(outputs.back());
        }

        void loadWeights(std::shared_ptr<weights::BinaryReader>& weightsReader) override
        {
            return;
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
