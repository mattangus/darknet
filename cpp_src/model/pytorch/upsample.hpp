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
        int _stride;

        torch::Tensor _forward(torch::Tensor x) {

            torch::IntArrayRef sizes = x.sizes();

            int64_t w, h;

            if (sizes.size() == 4)
            {
                w = sizes[2] * _stride;
                h = sizes[3] * _stride;

                x = torch::upsample_nearest2d(x, {w, h});
            }
            else if (sizes.size() == 3)
            {
                w = sizes[2] * _stride;
                x = torch::upsample_nearest1d(x, {w});
            }   	
            return x; 
        }
    public:
        Upsample(std::shared_ptr<params::UpsampleParams>& params, std::vector<int>& outputDepths) : DarknetModule("Upsample") {
            auto opt = torch::nn::UpsampleOptions();
            std::vector<double> v = {(double)params->stride, (double)params->stride};
            opt.scale_factor(v);
            _stride = params->stride;
            
            up = register_module("upsample", torch::nn::Upsample(opt));

            outputDepths.push_back(outputDepths.back());
        }
        ~Upsample() {}


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
