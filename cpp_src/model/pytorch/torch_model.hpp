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
        std::vector<std::shared_ptr<Yolo>> outputModules;
        std::vector<torch::Tensor> outputs;
        std::unordered_set<int> outputLayers;
    public:
        TorchModel(/* args */) {}
        ~TorchModel() {}

        void addModule(std::shared_ptr<DarknetModule>& mod, std::string& name)
        {
            modules.push_back(register_module(name, mod));
        }

        void addModule(std::shared_ptr<Yolo>& mod, std::string& name)
        {
            outputLayers.emplace(modules.size());
            modules.push_back(register_module(name, std::static_pointer_cast<DarknetModule>(mod)));
            outputModules.push_back(mod);
        }

        std::vector<torch::Tensor> forward(torch::Tensor input)
        {
            std::vector<torch::Tensor> ret;
            ret.reserve(outputLayers.size());

            outputs.clear();
            outputs.push_back(input);
            for(int i = 0; i < modules.size(); i++)
            {
                outputs.push_back(modules[i]->forward(outputs));
                if(outputLayers.count(i) > 0)
                    ret.push_back(outputs.back());
                // std::cout << i << " " << modules[i]->name << " " << outputs.back().sizes() << std::endl;
            }

            auto pkl = torch::pickle_save(outputs);
            std::ofstream out("out.pkl");
            out.write(&pkl[0], pkl.size());
            out.close();

            return ret;
        }

        torch::Tensor getOutput(int i)
        {
            return outputs[i];
        }

        std::vector<std::vector<Detection>> getBoxes(std::vector<torch::Tensor> outputs, std::vector<int> inputSize, float thresh)
        {
            std::vector<std::vector<Detection>> ret;
            for(int i = 0; i < outputs.size(); i++)
            {
                outputModules[i]->getBoxes(outputs[i], inputSize, thresh, ret);
            }

            return ret;
        }

        void loadWeights(std::shared_ptr<weights::BinaryReader>& weightsReader)
        {
            auto major = weightsReader->read<int32_t>();
            auto minor = weightsReader->read<int32_t>();
            auto revision = weightsReader->read<int32_t>();
            auto seen = weightsReader->read<int32_t>();
            weightsReader->read<int32_t>(); // extra

            for(auto& mod : modules)
            {
                mod->loadWeights(weightsReader);
            }

            auto numLeft = weightsReader->bytesLeft();
            assert(numLeft == 0);
        }

    };

} // namespace pytorch
} // namespace model
} // namespace darknet