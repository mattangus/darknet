#pragma once

#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/nms.h>
#include <torch/extension.h>
#include "params/layers.hpp"
#include "model/pytorch/dark_module.hpp"
#include "types/types.hpp"
#include <torch/jit.h>
#include <istream>
#include "utils/vis.hpp"

std::vector<std::string> names_ = {
"person",
"bicycle",
"car",
"motorbike",
"aeroplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"sofa",
"pottedplant",
"bed",
"diningtable",
"toilet",
"tvmonitor",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush",
};


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
        std::vector<float> masked_anchors;
    
        torch::Tensor predict_transform(torch::Tensor prediction, int inp_dim, int num_classes, torch::Device device)
        {
            int batch_size = prediction.size(0);
            int stride = floor(inp_dim / prediction.size(2));
            int grid_size = floor(inp_dim / stride);
            int bbox_attrs = 5 + num_classes;
            int num_anchors = masked_anchors.size()/2;

            for (size_t i = 0; i < masked_anchors.size(); i++)
            {
                masked_anchors[i] = masked_anchors[i]/stride;
            }
            torch::Tensor result = prediction.view({batch_size, bbox_attrs * num_anchors, grid_size * grid_size});
            result = result.transpose(1,2).contiguous();
            result = result.view({batch_size, grid_size*grid_size*num_anchors, bbox_attrs});
            
            result.select(2, 0).sigmoid_();
            result.select(2, 1).sigmoid_();
            result.select(2, 4).sigmoid_();

            auto grid_len = torch::arange(grid_size);

            std::vector<torch::Tensor> args = torch::meshgrid({grid_len, grid_len});

            torch::Tensor x_offset = args[1].contiguous().view({-1, 1});
            torch::Tensor y_offset = args[0].contiguous().view({-1, 1});

            // std::cout << "x_offset:" << x_offset << endl;
            // std::cout << "y_offset:" << y_offset << endl;

            x_offset = x_offset.to(device);
            y_offset = y_offset.to(device);

            auto x_y_offset = torch::cat({x_offset, y_offset}, 1).repeat({1, num_anchors}).view({-1, 2}).unsqueeze(0);
            result.slice(2, 0, 2).add_(x_y_offset);

            torch::Tensor anchors_tensor = torch::from_blob(masked_anchors.data(), {num_anchors, 2});
            //if (device != nullptr)
                anchors_tensor = anchors_tensor.to(device);
            anchors_tensor = anchors_tensor.repeat({grid_size*grid_size, 1}).unsqueeze(0);

            result.slice(2, 2, 4).exp_().mul_(anchors_tensor);
            result.slice(2, 5, 5 + num_classes).sigmoid_();
            result.slice(2, 0, 4).mul_(stride);

            return result;
        }

    public:
        Yolo(std::shared_ptr<params::YoloParams>& params, std::vector<int>& outputDepths) : DarknetModule("Yolo") {
            this->params = params;
            outputDepths.push_back(outputDepths.back());

            // std::vector<float> masked_anchors(params->mask.size() * 2);
            masked_anchors.resize(params->mask.size()*2);
            for(int i = 0; i < params->mask.size(); i++)
            {
                masked_anchors[i*2] = params->anchors[params->mask[i]*2];
                masked_anchors[i*2 + 1] = params->anchors[params->mask[i]*2 + 1];
            }

            anchors = register_buffer("anchors", torch::from_blob(masked_anchors.data(), {(long)masked_anchors.size()}, torch::kF32).reshape({-1, 2})).to(torch::kCUDA);
            
            std::cout << "anchors: " << std::endl << anchors << std::endl;
        }
        ~Yolo() {}

        torch::Tensor forward(std::vector<torch::Tensor>& outputs) override {
            return outputs.back();
        }

        torch::Tensor reshapeToAnchors(torch::Tensor ten) {
            auto shape = ten.sizes();
            int nachors = anchors.sizes()[0];
            torch::Tensor result = ten.reshape({shape[0], -1, shape[2] * shape[3]});
            result = result.transpose(1, 2).contiguous();
            
            return result.reshape({shape[0], shape[2] * shape[3], nachors, -1});
            // return ten.reshape({shape[0], nachors, -1, shape[2], shape[3]});
        }

        // void getBoxes(torch::Tensor prediction, std::vector<int> inputSize, float thresh, std::vector<std::vector<Detection>>& outputBoxes)
        // {
        //     int batch_size = prediction.size(0);
        //     int num_anchors = anchors.size(0);
        //     int bbox_attrs = prediction.size(1) / num_anchors;
        //     int grid_x = prediction.sizes()[2];
        //     int grid_y = prediction.sizes()[3];

        //     auto inSizes = torch::from_blob(inputSize.data(), {(long)inputSize.size()}, torch::kInt32).toType(torch::kF32).to(anchors.device());
        //     auto anchorsScaled = (anchors / inSizes);

        //     torch::Tensor result = prediction.view({batch_size, bbox_attrs * num_anchors, grid_x * grid_y});
        //     result = result.transpose(1,2).contiguous();
        //     result = result.view({batch_size, grid_x*grid_y*num_anchors, bbox_attrs});

        //     result.select(2, 0).sigmoid_();
        //     result.select(2, 1).sigmoid_();
        //     result.select(2, 4).sigmoid_();

        //     std::vector<torch::Tensor> xys = torch::meshgrid({torch::arange(grid_y), torch::arange(grid_x)});
        //     torch::Tensor x_offset = xys[1].contiguous().view({-1, 1});
        //     torch::Tensor y_offset = xys[0].contiguous().view({-1, 1});

        //     x_offset = x_offset.to(anchors.device());
        //     y_offset = y_offset.to(anchors.device());

        //     auto x_y_offset = torch::cat({x_offset, y_offset}, 1).repeat({1, num_anchors}).view({-1, 2}).unsqueeze(0);
        //     result.slice(2, 0, 2).add_(x_y_offset);
        //     auto anchors_tensor = anchors.repeat({grid_x*grid_y, 1}).unsqueeze(0);
        //     result.slice(2, 2, 4).exp_().mul_(anchors_tensor);
        //     result.slice(2, 5, bbox_attrs).sigmoid_();
   		//     result.slice(2, 0, 4).mul_(inSizes);
        // }

        void getBoxes(torch::Tensor output, std::vector<int> inputSize, float thresh, std::vector<std::vector<Detection>>& outputBoxes)
        {
            auto result = predict_transform(output, inputSize[0], 80, torch::kCUDA);

            auto box_xy = result.index(     {0, Slice(), Slice(0, 2)});
            auto box_hw = result.index(     {0, Slice(), Slice(2, 4)});
            auto objectivity = result.index({0, Slice(), Slice(4, 5)});
            auto class_prob = result.index( {0, Slice(), Slice(5, None)});

            std::cout << "max_dog: " << (class_prob*objectivity).index({Slice(), 16}).max() << std::endl;

            std::cout << "result: " << result.sizes() << std::endl;
            // int num_anchors = anchors.size(0);
            // int bbox_attrs = output.size(1) / num_anchors;
            // int numClass = bbox_attrs - 5;
            // int batch = output.sizes()[0];
            // int grid_x = output.sizes()[2];
            // int grid_y = output.sizes()[3];

            // output = reshapeToAnchors(output);

            // // {
            // //     std::vector<float> bin_in;
            // //     std::stringstream ss;
            // //     ss << "out_" << grid_x << ".npy";
            // //     std::cout << "loading from " << ss.str() << std::endl;
            // //     std::ifstream in(ss.str(), std::ios::binary);
            // //     float f;
            // //     while (in.read(reinterpret_cast<char*>(&f), sizeof(float)))
            // //         bin_in.push_back(f);
            // //     output = torch::from_blob(bin_in.data(), output.sizes(), torch::kF32).to(anchors.device());
            // // }
            
            // // torch::Tensor prediction = output.view({batch, bbox_attrs * num_anchors, grid_x * grid_y});
            // // prediction = prediction.transpose(1,2).contiguous();
            // // prediction = prediction.view({batch, grid_x*grid_y, num_anchors, bbox_attrs});

            // // std::cout << "output: " << output.sizes() << std::endl;
            // auto box_xy = output.index({Slice(), Slice(), Slice(), Slice(0, 2)});
            // auto box_hw = output.index({Slice(), Slice(), Slice(), Slice(2, 4)});
            // auto objectivity = output.index({Slice(), Slice(), Slice(), Slice(4, 5)});
            // auto class_prob = output.index({Slice(), Slice(), Slice(), Slice(5, None)});
            // // std::cout << "box_xy: " << box_xy.sizes() << std::endl;
            // // std::cout << "box_hw: " << box_hw.sizes() << std::endl;
            // // std::cout << "objectivity: " << objectivity.sizes() << std::endl;
            // // std::cout << "class_prob: " << class_prob.sizes() << std::endl;

            // box_xy = torch::sigmoid(box_xy);
            // if(params->scale_x_y != 1)
            // {
            //     float alpha = params->scale_x_y;
            //     float beta = -0.5f*(params->scale_x_y - 1);
            //     box_xy = (box_xy * alpha) + beta;
            // }

            // objectivity = torch::sigmoid(objectivity);
            // class_prob = torch::sigmoid(class_prob);

            // // assert(output.dim() == 5);
            // assert(inputSize.size() == 2);
            
            // int start = 0;
            // auto grid_xy = torch::meshgrid({torch::arange(0, grid_x, torch::kF32), torch::arange(0, grid_y, torch::kF32)});
            // auto grid = torch::stack({grid_xy[1], grid_xy[0]}, -1).to(anchors.device()).view({1, grid_x*grid_y, 1, 2});

            // auto inSizes = torch::from_blob(inputSize.data(), {(long)inputSize.size()}, torch::kInt32).toType(torch::kF32).to(anchors.device()).toType(torch::kF32);

            // std::vector<int> gridS = {grid_x, grid_y};
            // auto gridSizes = torch::from_blob(gridS.data(), {2}, torch::kInt32).toType(torch::kF32).to(anchors.device());
            // gridSizes = gridSizes.view({1, 1, 1, 2});

            // // std::cout << "anchors:" << anchors << std::endl;
            // // std::cout << "inSizes:" << inSizes << std::endl;

            // auto anchorsScaled = (anchors / inSizes);

            // std::set<std::string> care = {"dog", "car", "bicycle", "truck"};

            // // vis::imshow("gridx", grid.view({grid_x, grid_y, 2}).index({Slice(), Slice(), 0}), 4);
            // // vis::imshow("gridy", grid.view({grid_x, grid_y, 2}).index({Slice(), Slice(), 1}), 4);
            // vis::imshow("objectivity", objectivity.view({grid_x, grid_y, num_anchors}).index({Slice(), Slice(), 0}), 4);
            // vis::imshow("box_x", box_xy.view({grid_x, grid_y, num_anchors, 2}).index({Slice(), Slice(), 0, 0}), 4);
            // vis::imshow("box_y", box_xy.view({grid_x, grid_y, num_anchors, 2}).index({Slice(), Slice(), 0, 1}), 4);
            // vis::imshow("box_h", box_hw.view({grid_x, grid_y, num_anchors, 2}).index({Slice(), Slice(), 0, 0}), 4);
            // vis::imshow("box_w", box_hw.view({grid_x, grid_y, num_anchors, 2}).index({Slice(), Slice(), 0, 1}), 4);

            // std::cout << "max obj: " << objectivity.max() << std::endl;

            // auto temp_class = (objectivity*class_prob).view({grid_x, grid_y, num_anchors, numClass});
            // for(int i = 0; i < names_.size(); i++)
            //     if(care.count(names_[i]) > 0)
            //     {
            //         vis::imshow(names_[i], temp_class.index({Slice(), Slice(), 0, i}), 4);
            //         std::cout << names_[i] << ": " << temp_class.index({Slice(), Slice(), Slice(), i}).max() << std::endl;
            //     }
            // cv::waitKey();

            // // std::cout << "anchorsScaled:" << anchorsScaled << std::endl;
            // anchorsScaled = anchorsScaled.view({1, 1, -1, 2});
            // // std::cout << "anchorsScaled: " << anchorsScaled.sizes() << std::endl;

            // box_hw = torch::exp(box_hw) * anchorsScaled;

            // // std::cout << "grid: " << grid.sizes() << std::endl;
            // // std::cout << "gridSizes: " << gridSizes << std::endl;
            // // std::cout << "gridSizes: " << gridSizes.sizes() << std::endl;

            // // std::cout << class_prob.sizes() << std::endl;

            // box_xy = (box_xy + grid) / gridSizes;
            // // box_xy = (grid) / gridSizes;
            // // std::cout << box_xy << std::endl;
            
            // // auto mask = objectivity > thresh;
            // // std::cout << "mask: " << mask.sizes() << std::endl;
            // // // vis::imshow("mask", mask.index({0,0,0}).to(torch::kFloat32));
            // // mask = mask.reshape({batch, -1});
            // // box_xy = box_xy.permute({0,1,3,4,2}).reshape({batch, -1, 2}).index(mask).reshape({batch, -1, 2});
            // // box_hw = box_hw.permute({0,1,3,4,2}).reshape({batch, -1, 2}).index(mask).reshape({batch, -1, 2});
            // // objectivity = objectivity.reshape({batch, -1, 1}).index(mask);
            // // class_prob = class_prob.permute({0,1,3,4,2}).reshape({batch, -1, numClass}).index(mask).reshape({batch, -1, numClass}) * objectivity;
            // // class_prob = class_prob*objectivity;

            // // auto pkl = torch::pickle_save(output);
            // // std::ofstream out("out.pkl");
            // // out.write(&pkl[0], pkl.size());
            // // out.close();
            // // box_xy = box_xy.permute({0,1,3,4,2});
            // // box_hw = box_hw.permute({0,1,3,4,2});
            // // std::cout << "permute boxes: " << box_xy.sizes() << std::endl;

            // auto box_xy_cpu = box_xy.detach().reshape({batch, -1, 2}).to(torch::kCPU);
            // auto box_hw_cpu = box_hw.detach().reshape({batch, -1, 2}).to(torch::kCPU);
            // auto class_prob_cpu = (class_prob*objectivity).detach().reshape({batch, -1, numClass}).to(torch::kCPU);

            // auto maxes = class_prob_cpu.index({0, Slice(), 16}).max(0);
            // float max_val = std::get<0>(maxes).item<float>();
            // int max_loc = std::get<1>(maxes).item<float>();

            // // std::cout << "max dog cpu: " << max_val << std::endl;
            // // std::cout << "max dog loc: " << max_loc << std::endl;

            // // std::cout << "class_prob_cpu: " << class_prob_cpu.sizes() << std::endl;

            // int numDet = class_prob_cpu.sizes()[1];

            // if(outputBoxes.size() != batch)
            //     outputBoxes.resize(batch);

            // // float max_score = -1;
            // // float max_dog = -1;
            // for(int b = 0; b < batch; b++)
            // {
            //     outputBoxes[b].reserve(numDet);
            //     for(int i = 0; i < numDet; i++)
            //     {
            //         auto cur_scores = class_prob_cpu.index({b, i});
            //         std::vector<float> scores(cur_scores.numel());
            //         std::memcpy((void *) scores.data(), cur_scores.data_ptr(), sizeof(float) * cur_scores.numel());

            //         auto x = box_xy_cpu[b][i][0];
            //         auto y = box_xy_cpu[b][i][1];
            //         auto h = box_hw_cpu[b][i][0];
            //         auto w = box_hw_cpu[b][i][1];

            //         // max_dog = std::max(max_dog, scores[16]);

            //         BoundingBox box(x.item<float>(), y.item<float>(), w.item<float>(), h.item<float>());

            //         // auto result = std::max_element(scores.begin(), scores.end());
            //         // int ind = std::distance(scores.begin(), result);
            //         // auto top_score = *result;
            //         // if(top_score>max_score)
            //         //     max_score = top_score;
            //         // if(top_score > 0.8)
            //         // {
            //         //     std::cout << names_[ind] << ": " << top_score << std::endl;
            //         // }
            //         // if(i == max_loc)
            //         // {
            //         //     std::cout << "torch: " << cur_scores << std::endl;
            //         //     std::cout << "vector: ";
            //         //     for(int j = 0; j < scores.size(); j++)
            //         //         std::cout << scores[j] << ", ";
            //         //     std::cout << std::endl;
            //         // }

            //         outputBoxes[b].push_back(Detection(box, scores));
            //     }
            // }

            // // std::cout << "max score: " << max_score << std::endl;
            // // std::cout << "max dog: " << max_dog << std::endl;

            // // std::cout << "#######################################" << std::endl << std::endl;
        }

        void loadWeights(std::shared_ptr<weights::BinaryReader>& weightsReader) override
        {
            return;
        }
    };
} // namespace torch
} // namespace model
} // namespace darknet
