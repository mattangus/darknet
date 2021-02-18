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
    using namespace torch::indexing;
    class Yolo : public DarknetModule
    {
    private:
    std::shared_ptr<params::YoloParams> params;
    public:
        Yolo(std::shared_ptr<params::YoloParams>& params, std::vector<int>& outputDepths) : DarknetModule("Yolo") {
            this->params = params;
            outputDepths.push_back(outputDepths.back());
        }
        ~Yolo() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            auto input = outputs.back();
            auto box_xy = input.index({Slice(), Slice(0, 2)});
            auto box_hw = input.index({Slice(), Slice(2, 4)});
            auto objectivity = torch::sigmoid(input.index({Slice(), Slice(4, 5)}));
            auto class_prob = torch::sigmoid(input.index({Slice(), Slice(5, None)}));
            
            box_xy = torch::sigmoid(box_xy);
            if(params->scale_x_y != 1)
            {
                float alpha = params->scale_x_y;
                float beta = -0.5f*(params->scale_x_y - 1);
                box_xy = (box_xy * alpha) + beta;
            }

            return torch::cat({box_xy, box_hw, objectivity, class_prob}, 1);
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
