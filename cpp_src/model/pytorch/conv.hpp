#pragma once

#include <torch/torch.h>
#include "params/layers.hpp"
#include "model/pytorch/dark_module.hpp"
#include "model/pytorch/activation.hpp"

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
        bool useBn, useAct;
        // std::shared_ptr<ActivationModule> act;
        ActivationType actType;
    public:
        Conv2d(std::shared_ptr<params::ConvParams>& params, std::vector<int>& outputDepths) : DarknetModule("Conv") {
            auto opt = torch::nn::Conv2dOptions(outputDepths.back(), params->filters, params->kernelSize);
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
                bn_opt.affine(true);
                bn = register_module("bn", torch::nn::BatchNorm2d(bn_opt));
            }

            // std::stringstream ss;
            // ss << params->activation;
            actType = params->activation;
            // act = register_module(ss.str(), getActivation(params->activation));

            outputDepths.push_back(params->filters);
        }
        ~Conv2d() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            auto input = outputs.back();
            auto output = conv->forward(input);
            if(useBn)
                output = bn->forward(output);

            return activate(output, actType);
            // return act->forward(output);
        }

        void loadWeights(std::shared_ptr<weights::BinaryReader>& weightsReader) override
        {

            if(useBn)
            {
                loadIntoTensor(&(bn->bias), weightsReader);
                loadIntoTensor(&(bn->weight), weightsReader);
                loadIntoTensor(&(bn->running_mean), weightsReader);
                loadIntoTensor(&(bn->running_var), weightsReader);
            }
            else
            {
                loadIntoTensor(&(conv->bias), weightsReader);
            }
            loadIntoTensor(&(conv->weight), weightsReader);
        }

    };
} // namespace torch
} // namespace model
} // namespace darknet
