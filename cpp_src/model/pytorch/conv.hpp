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
    class Conv2d : public DarknetModule
    {
    private:
        /* data */
        torch::nn::BatchNorm2d bn{nullptr};
        torch::nn::Conv2d conv{nullptr};
        bool useBn;
    public:
        Conv2d(std::shared_ptr<params::ConvParams>& params, int* input_depth) {
            auto opt = torch::nn::Conv2dOptions(*input_depth, params->filters, params->kernelSize);
            opt.groups(params->groups);
            opt.stride({params->strides.first, params->strides.second});
            opt.padding(params->padding);
            if(params->batch_normalize)
                opt.bias(false);
            else
                opt.bias(true);
            opt.padding_mode(torch::kZeros);
            
            conv = register_module("conv", torch::nn::Conv2d(opt));

            useBn = params->batch_normalize;
            if(params->batch_normalize)
            {
                auto bn_opt = torch::nn::BatchNorm2dOptions(params->filters);
                bn_opt.eps(0.00001); // line 499 of convolutional_kernels.cu
                bn_opt.momentum(0.01); // line 496 of same file
                bn_opt.affine(false);
                bn = register_module("bn", torch::nn::BatchNorm2d(bn_opt));
            }

            *input_depth = params->filters;
        }
        ~Conv2d() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            auto input = outputs.back();
            auto output = conv->forward(input);
            if(useBn)
                output = bn->forward(output);

            return output;
        }

    };
} // namespace torch
} // namespace model
} // namespace darknet
