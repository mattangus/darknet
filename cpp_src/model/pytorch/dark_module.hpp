#pragma once

#include <torch/torch.h>
#include "params/layers.hpp"
#include "weights/binary_reader.hpp"

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
        std::string name;
        DarknetModule(std::string name) : name(name) {
            
            
        }
        ~DarknetModule() {}

        virtual torch::Tensor forward(std::vector<torch::Tensor>& outputs) = 0;
        virtual void loadWeights(std::shared_ptr<weights::BinaryReader>& weightsReader) = 0;

        void loadIntoTensor(torch::Tensor* t, std::shared_ptr<weights::BinaryReader>& weightsReader)
        {
            torch::NoGradGuard guard;
            auto tw = weightsReader->readN<float>(t->numel());

            auto tt = torch::from_blob(tw.data(), t->sizes());
            t->copy_(tt.clone().detach());
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
