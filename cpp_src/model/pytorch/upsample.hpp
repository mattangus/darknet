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
        std::vector<double> strides;

    public:
        Upsample(std::shared_ptr<params::UpsampleParams>& params, std::vector<int>& outputDepths) : DarknetModule("Upsample") {
            auto opt = torch::nn::UpsampleOptions();
            std::vector<double> v = {(double)params->stride, (double)params->stride};
            opt.scale_factor(v);
            opt.mode(torch::kNearest);
            
            up = register_module("upsample", torch::nn::Upsample(opt));

            outputDepths.push_back(outputDepths.back());
        }
        ~Upsample() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            // auto sizes = outputs.back().sizes();
            // int64_t w, h;
            // if (sizes.size() == 4)
            // {
            //     w = sizes[2] * strides[0];
            //     h = sizes[3] * strides[1];

            //     auto x = torch::upsample_nearest2d(outputs.back(), {w, h});
            //     outputs.push_back(x);
            // }
            // else if (sizes.size() == 3)
            // {
            //     w = sizes[2] * strides[0];
            //     auto x = torch::upsample_nearest1d(outputs.back(), {w});
            //     outputs.push_back(x);
            // }   	
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
