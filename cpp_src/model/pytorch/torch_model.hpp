#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <unordered_set>
#include "model/pytorch/dark_module.hpp"

namespace darknet
{
namespace model
{
namespace pytorch
{

    class TorchModel : public torch::nn::Module
    {
    private:
        std::vector<std::shared_ptr<DarknetModule>> modules;
        std::vector<torch::Tensor> outputs;
    public:
        TorchModel(/* args */) {}
        ~TorchModel() {}

        void addModule(std::shared_ptr<DarknetModule>& mod, std::string& name)
        {
            modules.push_back(register_module(name, mod));
        }

        torch::Tensor forward(torch::Tensor input)
        {
            outputs.clear();
            outputs.push_back(input);
            for(int i = 0; i < modules.size(); i++)
            {
                outputs.push_back(modules[i]->forward(outputs));
                // std::cout << i << " " << modules[i]->name << " " << outputs.back().sizes() << std::endl;
            }

            return outputs.back();
        }
    };

} // namespace pytorch
} // namespace model
} // namespace darknet