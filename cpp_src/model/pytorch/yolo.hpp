#pragma once

#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/nms.h>
#include "params/layers.hpp"
#include "model/pytorch/dark_module.hpp"
#include "types/types.hpp"


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
    torch::Tensor anchors;
    public:
        Yolo(std::shared_ptr<params::YoloParams>& params, std::vector<int>& outputDepths) : DarknetModule("Yolo") {
            this->params = params;
            outputDepths.push_back(outputDepths.back());

            std::vector<float> masked_anchors(params->mask.size() * 2);
            for(int i = 0; i < params->mask.size(); i++)
            {
                masked_anchors[i*2] = params->anchors[params->mask[i]*2];
                masked_anchors[i*2 + 1] = params->anchors[params->mask[i]*2 + 1];
            }

            anchors = register_buffer("anchors", torch::from_blob(masked_anchors.data(), {(long)masked_anchors.size()}, torch::kF32).reshape({-1, 2}));
            
            // std::cout << "anchors: " << std::endl << anchors << std::endl;
        }
        ~Yolo() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            auto input = outputs.back();
            auto box_xy = input.index({Slice(), Slice(0, 2)});
            auto box_hw = input.index({Slice(), Slice(2, 4)});
            auto obj_and_class = torch::sigmoid(input.index({Slice(), Slice(4, None)}));
            
            box_xy = torch::sigmoid(box_xy);
            if(params->scale_x_y != 1)
            {
                float alpha = params->scale_x_y;
                float beta = -0.5f*(params->scale_x_y - 1);
                box_xy = (box_xy * alpha) + beta;
            }

            auto ret = torch::cat({box_xy, box_hw, obj_and_class}, 1);
            auto shape = ret.sizes();
            int nachors = anchors.sizes()[0];
            return ret.reshape({shape[0], nachors, -1, shape[2], shape[3]});
        }

        std::vector<Detection> getBoxes(torch::Tensor output, std::vector<int> inputSize, float thresh)
        {
            assert(output.dim() == 5);
            assert(inputSize.size() == 2);

            int batch = output.sizes()[0];
            int grid_x = output.sizes()[3];
            int grid_y = output.sizes()[4];
            
            int start = 0;
            auto grid_xy = torch::meshgrid({torch::range(0, grid_x-1, torch::kF32).to(anchors.device()), torch::range(0, grid_y-1, torch::kF32).to(anchors.device())});
            auto grid = torch::stack(grid_xy, 0).unsqueeze(0).unsqueeze(0);

            auto box_xy = output.index({Slice(), Slice(), Slice(0, 2)});
            auto box_hw = torch::exp(output.index({Slice(), Slice(), Slice(2, 4)}));
            auto objectivity = output.index({Slice(), Slice(), Slice(4, 5)});
            auto class_prob = output.index({Slice(), Slice(), Slice(5, None)});

            int numClass = class_prob.sizes()[2];

            auto inSizes = torch::from_blob(inputSize.data(), {(long)inputSize.size()}, torch::kInt32).toType(torch::kF32).to(anchors.device());

            std::vector<int> gridS = {grid_x, grid_y};
            auto gridSizes = torch::from_blob(gridS.data(), {2}, torch::kInt32).toType(torch::kF32).to(anchors.device());
            gridSizes = gridSizes.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0);

            auto anchorsScaled = (anchors / inSizes).unsqueeze(-1).unsqueeze(-1).unsqueeze(0);

            // std::cout << anchorsScaled.sizes() << std::endl;

            box_hw = box_hw * anchorsScaled;

            // std::cout << grid.sizes() << std::endl;
            // std::cout << gridSizes.sizes() << std::endl;

            std::cout << class_prob.sizes() << std::endl;

            box_xy = (box_xy + grid) / gridSizes;
            
            objectivity = objectivity.reshape({batch, -1});
            auto mask = objectivity > thresh;
            box_xy = box_xy.permute({0,1,3,4,2}).reshape({batch, -1, 2}).index(mask);
            box_hw = box_hw.permute({0,1,3,4,2}).reshape({batch, -1, 2}).index(mask);
            objectivity = objectivity.index(mask).unsqueeze(-1);
            class_prob = class_prob.permute({0,1,3,4,2}).reshape({batch, -1, numClass}).index(mask) * objectivity;

            
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
