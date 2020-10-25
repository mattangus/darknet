
#include "atensor/tensor_cpu.hpp"

#include <gtest/gtest.h>
// TEST(DarkNetTest, CreatingObjects) {
//     std::string act = "logistic";
//     auto layer_1 = darknet::layer::Activation(nullptr, act);
// }

using namespace darknet::tensor;

TEST(DarkentTensor, TestCreateCPUTensor)
{
    TensorShape shape({9, 3});
    
    CpuTensor<float> matrix1();
    auto matrix2 = std::make_shared<CpuTensor<float>>(shape);
    auto matrix3 = std::make_shared<CpuTensor<float>>(shape);
    auto temp = std::static_pointer_cast<TensorBase<float>>(matrix3);
    
    matrix2->copyTo(temp);
}


// TEST(DarkentTensor, GPUActivation)
// {   
//     darknet::tensor::TensorShape shape({9, 15});
//     auto matrix1 = std::make_shared<darknet::tensor::Tensor<float, darknet::DeviceType::GPU>> (shape);
//     std::vector<float> inputs(shape.numElem());
//     for(int i = 0; i < shape.numElem(); i++)
//         inputs[i] = i / 10;
//     matrix1->fromArray(inputs);

//     auto inp = std::make_shared<darknet::layer::Input<darknet::DeviceType::GPU>>(matrix1);
//     inp->init();
//     auto inpl = std::static_pointer_cast<darknet::layer::Layer<darknet::DeviceType::GPU>>(inp);
//     auto act = std::make_shared<darknet::layer::Activation<darknet::DeviceType::GPU>>(inpl);
//     std::shared_ptr<darknet::network::NetworkState> s = nullptr;
//     act->forward(s);
// }


int main(int argc, char **argv) {
    // register_handlers();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}