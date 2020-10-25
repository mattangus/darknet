
#include "atensor/tensor_cpu.hpp"

#include <gtest/gtest.h>
// TEST(DarkNetTest, CreatingObjects) {
//     std::string act = "logistic";
//     auto layer_1 = darknet::layer::Activation(nullptr, act);
// }

using namespace darknet::tensor;

TEST(DarkentTensor, TestCPUCopy)
{
    TensorShape shape({9, 3});
    
    CpuTensor<float> matrix1();
    auto matrix2 = std::make_shared<CpuTensor<float>>(shape);
    auto matrix3 = std::make_shared<CpuTensor<float>>(shape);
    auto temp = std::static_pointer_cast<TensorBase<float>>(matrix3);

    matrix2->copyTo(temp);
}

int main(int argc, char **argv) {
    // register_handlers();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}