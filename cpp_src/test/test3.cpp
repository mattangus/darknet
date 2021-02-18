#include<torch/torch.h>
#include <gtest/gtest.h>
#include <chrono>

#include "parser/torchBuilder.hpp"
#include "parser/cfgparser.hpp"
#include "parser/TextReader.hpp"

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

    auto mReader = std::static_pointer_cast<reader>(std::make_shared<TextReader>("cfg/yolov4.cfg"));
    auto builder = std::make_shared<torchBuilder>(3);
    auto _builder = std::static_pointer_cast<NetworkBuilder>(builder);
    auto parser = std::make_shared<CfgParser>();
    
    parser->parseConfig(mReader, _builder);

    torch::Device device(torch::kCUDA);

    auto model = builder->getModel();
    model.to(device);
    std::cout << model << std::endl;
    std::cout << "cuda: " << torch::cuda::is_available() << std::endl;
    std::cout << "cudnn: " << torch::cuda::cudnn_is_available() << std::endl;
    std::cout << "is_cuda: " << model.parameters().back().is_cuda() << std::endl;
    auto input = torch::rand({1, 3, 512, 512}, device);
    std::cout << "input: " << input.device() << std::endl;
    int n = 300;
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
}