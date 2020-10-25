
#include "tensor/tensor.hpp"
#include "tensor/tensor_cpu.hpp"
#include "tensor/tensor_gpu.hpp"
#include "tensor/tensor_shape.hpp"
#include "utils/signal_handler.hpp"
#include "layer/input.hpp"
#include "layer/activation.hpp"

#include <gtest/gtest.h>
// TEST(DarkNetTest, CreatingObjects) {
//     std::string act = "logistic";
//     auto layer_1 = darknet::layer::Activation(nullptr, act);
// }

using namespace darknet;
using namespace darknet::tensor;

TEST(DarkentTensor, TestCreateCPUTensor)
{
    TensorShape shape({9, 3});
    
    Tensor<float, DeviceType::CPU> matrix1();
    Tensor<float, DeviceType::CPU> matrix2(shape);
}

TEST(DarkentTensor, TestCreateGPUTensor)
{
    TensorShape shape({10, 10});
    
    Tensor<float, DeviceType::GPU> matrix1();
    Tensor<float, DeviceType::GPU> matrix2(shape);
}

TEST(DarkentTensor, TestCopyCPU)
{
    TensorShape shape({9, 3});
    Tensor<float, DeviceType::CPU> matrix1(shape);
    Tensor<float, DeviceType::CPU> matrix2(shape);
    Tensor<float, DeviceType::CPU> matrix3 = matrix1.copy();
    matrix1.copyTo(matrix2);
}

TEST(DarkentTensor, TestCopyGPU)
{
    TensorShape shape({4, 3});
    Tensor<float, DeviceType::GPU> matrix1(shape);
    Tensor<float, DeviceType::GPU> matrix2(shape);
    Tensor<float, DeviceType::GPU> matrix3 = matrix1.copy();
    matrix1.copyTo(matrix2);
}


TEST(DarkentTensor, ApplyCPU)
{
    TensorShape shape({9, 15});
    Tensor<float, DeviceType::CPU> matrix1(shape);
    matrix1.apply<ActivationType::RELU>();
}

TEST(DarkentTensor, ApplyGPU)
{
    TensorShape shape({9, 20});
    Tensor<float, DeviceType::GPU> matrix12(shape);
    matrix12.apply<ActivationType::RELU>();
}

TEST(DarkentTensor, CPUInput)
{
    TensorShape shape({9, 15});
    std::shared_ptr<Tensor<float, DeviceType::CPU>> matrix1(new Tensor<float, DeviceType::CPU>(shape));
    std::vector<float> inputs(shape.numElem());
    for(int i = 0; i < shape.numElem(); i++)
        inputs[i] = i / 10;
    matrix1->fromArray(inputs);

    layer::Input<DeviceType::CPU> inp(matrix1);
    inp.init();
}

TEST(DarkentTensor, GPUInput)
{   
    TensorShape shape({9, 15});
    std::shared_ptr<Tensor<float, DeviceType::GPU>> matrix1(new Tensor<float, DeviceType::GPU>(shape));
    std::vector<float> inputs(shape.numElem());
    for(int i = 0; i < shape.numElem(); i++)
        inputs[i] = i / 10;
    matrix1->fromArray(inputs);

    layer::Input<DeviceType::GPU> inp(matrix1);
}

TEST(DarkentTensor, CPUActivation)
{
    TensorShape shape({9, 15});
    auto matrix1 = std::make_shared<Tensor<float, DeviceType::CPU>> (shape);
    std::vector<float> inputs(shape.numElem());
    for(int i = 0; i < shape.numElem(); i++)
        inputs[i] = i / 10;
    matrix1->fromArray(inputs);

    auto inp = std::make_shared<layer::Input<DeviceType::CPU>>(matrix1);
    inp->init();

    auto inpl = std::static_pointer_cast<layer::Layer<DeviceType::CPU>>(inp);
    auto act = std::make_shared<layer::Activation<DeviceType::CPU>>(inpl, ActivationType::RELU);
    std::shared_ptr<network::NetworkState> s = nullptr;
    act->forward(s);
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