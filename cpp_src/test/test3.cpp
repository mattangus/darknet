    #include<torch/torch.h>
#include <gtest/gtest.h>

#include "parser/torchBuilder.hpp"
#include "parser/cfgparser.hpp"
#include "parser/TextReader.hpp"

// TEST(DarkNetTest, CreatingObjects) {
//     std::string act = "logistic";
//     auto layer_1 = darknet::layer::Activation(nullptr, act);
// }

using namespace darknet::parser;

TEST(DarkentTensor, TestCreateGPUTensor)
{
    auto opt = torch::TensorOptions();
    torch::Tensor tensor = torch::rand({2, 3}).to(torch::kCUDA);
    // std::cout << tensor << std::endl;
}


TEST(DarkentTensor, TestCreateCPUTensor)
{
    torch::Tensor tensor = torch::rand({2, 3});
    // std::cout << tensor << std::endl;
}

TEST(DarkentParser, Test)
{
    auto mReader = std::static_pointer_cast<reader>(std::make_shared<TextReader>("../cfg/yolov4.cfg"));
    auto builder = std::static_pointer_cast<NetworkBuilder>(std::make_shared<torchBuilder>());
    auto parser = std::make_shared<CfgParser>();
    
    parser->parseConfig(mReader, builder);
}


int main(int argc, char **argv) {
    // register_handlers();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}