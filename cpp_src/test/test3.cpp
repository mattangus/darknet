#include<torch/torch.h>

#include <gtest/gtest.h>
// TEST(DarkNetTest, CreatingObjects) {
//     std::string act = "logistic";
//     auto layer_1 = darknet::layer::Activation(nullptr, act);
// }

TEST(DarkentTensor, TestCreateCPUTensor)
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
}


int main(int argc, char **argv) {
    // register_handlers();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}