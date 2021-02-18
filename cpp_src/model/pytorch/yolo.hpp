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
    class Yolo : public DarknetModule
    {
    private:
        torch::nn::Identity up{nullptr};

    public:
        Yolo(std::shared_ptr<params::YoloParams>& params, std::vector<int>& outputDepths) : DarknetModule("Yolo") {

            outputDepths.push_back(outputDepths.back());
        }
        ~Yolo() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            return outputs.back();
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
