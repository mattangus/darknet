#pragma once

#include <torch/torch.h>
#include "params/layers.hpp"

namespace darknet
{
namespace model
{
namespace pytorch
{
    class DarknetModule : public torch::nn::Module
    {
    private:
        /* data */
        
    public:
        DarknetModule() {
            
            
        }
        ~DarknetModule() {}

        virtual torch::Tensor forward(std::vector<torch::Tensor>& outputs) = 0;
    };
} // namespace torch
} // namespace model
} // namespace darknet
