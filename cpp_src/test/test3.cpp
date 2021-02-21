#include<torch/torch.h>
#include <gtest/gtest.h>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "parser/torchBuilder.hpp"
#include "parser/cfgparser.hpp"
#include "parser/TextReader.hpp"
#include "weights/binary_reader.hpp"

// TEST(DarkNetTest, CreatingObjects) {
//     std::string act = "logistic";
//     auto layer_1 = darknet::layer::Activation(nullptr, act);
// }

using namespace darknet::parser;

// TEST(DarkentTensor, TestCreateGPUTensor)
// {
//     auto opt = torch::TensorOptions();
//     torch::Tensor tensor = torch::rand({2, 3}).to(torch::kCUDA);
//     // std::cout << tensor << std::endl;
// }


// TEST(DarkentTensor, TestCreateCPUTensor)
// {
//     torch::Tensor tensor = torch::rand({2, 3});
//     // std::cout << tensor << std::endl;
// }

// TEST(DarkentParser, Test)
// {
//     auto mReader = std::static_pointer_cast<reader>(std::make_shared<TextReader>("cfg/yolov4.cfg"));
//     auto builder = std::static_pointer_cast<NetworkBuilder>(std::make_shared<torchBuilder>(3));
//     auto parser = std::make_shared<CfgParser>();
    
//     parser->parseConfig(mReader, builder);
// }


int main(int argc, char **argv) {
    // register_handlers();
    // testing::InitGoogleTest(&argc, argv);
    // return RUN_ALL_TESTS();

    torch::Device device(torch::kCUDA);

    std::string cfgPath = "cfg/yolov4.cfg";
    std::string weightsPath = "original_weights/yolov4/yolov4.weights";
    std::string inputPath = "data/dog.jpg";
    auto frame = cv::imread(inputPath, -1);
    int frame_w = frame.cols;
    int frame_h = frame.rows;
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
    auto input = torch::from_blob(frame.data, {1, frame_h, frame_w, 3});
    input = input.permute({0, 3, 1, 2}).to(device);

    if(argc == 3)
    {
        cfgPath = argv[1];
        weightsPath = argv[2];
    }

    auto mReader = std::static_pointer_cast<reader>(std::make_shared<TextReader>(cfgPath));
    auto builder = std::make_shared<torchBuilder>(3);
    auto _builder = std::static_pointer_cast<NetworkBuilder>(builder);
    auto parser = std::make_shared<CfgParser>();
    
    parser->parseConfig(mReader, _builder);
    auto model = builder->getModel();

    auto br = std::make_shared<darknet::weights::BinaryReader>(weightsPath);

    model.loadWeights(br);
    model.to(device);
    // std::cout << model << std::endl;
    // std::cout << "cuda: " << torch::cuda::is_available() << std::endl;
    // std::cout << "cudnn: " << torch::cuda::cudnn_is_available() << std::endl;
    // std::cout << "is_cuda: " << model.parameters().back().is_cuda() << std::endl;
    // auto input = torch::rand({1, 3, 512, 512}, device);
    // std::cout << "input: " << input.device() << std::endl;
    int n = 10;
    for(int i = 0; i < 5; i++) // burn in
        model.forward(input);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n; i++)
    {
        auto res = model.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << (diff.count() / n) << std::endl;

    auto res = model.forward(input);
    for(int i = 0; i < res.size(); i++)
    {
        std::cout << res[i].sizes() << std::endl;
    }
    std::cout << std::endl;
    auto boxes = model.getBoxes(res, {frame_h, frame_w}, 0.01);

    for(int i = 0; i < boxes[0].size(); i++)
    {
        auto b = boxes[0][i].bbox;
        cv::rectangle(frame, cv::Point((b.cx - b.w/2)*frame_w, (b.cy - b.h/2)*frame_h), cv::Point((b.cx + b.w/2)*frame_w, (b.cy + b.h/2)*frame_h), {0, 255, 0});
    }

    cv::imshow("test", frame);
    cv::waitKey();
}