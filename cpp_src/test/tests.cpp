
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

TEST(DarkentTensor, TestCreateCPUTensor)
{
    darknet::tensor::TensorShape shape({9, 3});
    
    darknet::tensor::Tensor<float, darknet::DeviceType::CPUDEVICE> matrix1();
    darknet::tensor::Tensor<float, darknet::DeviceType::CPUDEVICE> matrix2(shape);
}

TEST(DarkentTensor, TestCreateGPUTensor)
{
    darknet::tensor::TensorShape shape({10, 10});
    
    darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE> matrix1();
    darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE> matrix2(shape);
}

TEST(DarkentTensor, TestCopyCPU)
{
    darknet::tensor::TensorShape shape({9, 3});
    darknet::tensor::Tensor<float, darknet::DeviceType::CPUDEVICE> matrix1(shape);
    darknet::tensor::Tensor<float, darknet::DeviceType::CPUDEVICE> matrix2(shape);
    darknet::tensor::Tensor<float, darknet::DeviceType::CPUDEVICE> matrix3 = matrix1.copy();
    matrix1.copyTo(matrix2);
}

TEST(DarkentTensor, TestCopyGPU)
{
    darknet::tensor::TensorShape shape({4, 3});
    darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE> matrix1(shape);
    darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE> matrix2(shape);
    darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE> matrix3 = matrix1.copy();
    matrix1.copyTo(matrix2);
}


TEST(DarkentTensor, ApplyCPU)
{
    darknet::tensor::TensorShape shape({9, 15});
    darknet::tensor::Tensor<float, darknet::DeviceType::CPUDEVICE> matrix1(shape);
    matrix1.apply<darknet::ActivationType::RELU>();
}

TEST(DarkentTensor, ApplyGPU)
{
    darknet::tensor::TensorShape shape({9, 20});
    darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE> matrix12(shape);
    matrix12.apply<darknet::ActivationType::RELU>();
}

TEST(DarkentTensor, CPUInput)
{
    darknet::tensor::TensorShape shape({9, 15});
    std::shared_ptr<darknet::tensor::Tensor<float, darknet::DeviceType::CPUDEVICE>> matrix1(new darknet::tensor::Tensor<float, darknet::DeviceType::CPUDEVICE>(shape));
    std::vector<float> inputs(shape.numElem());
    for(int i = 0; i < shape.numElem(); i++)
        inputs[i] = i / 10;
    matrix1->fromArray(inputs);

    darknet::layer::Input<darknet::DeviceType::CPUDEVICE> inp(matrix1);
    inp.init();
}

TEST(DarkentTensor, GPUInput)
{   
    darknet::tensor::TensorShape shape({9, 15});
    std::shared_ptr<darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE>> matrix1(new darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE>(shape));
    std::vector<float> inputs(shape.numElem());
    for(int i = 0; i < shape.numElem(); i++)
        inputs[i] = i / 10;
    matrix1->fromArray(inputs);

    darknet::layer::Input<darknet::DeviceType::GPUDEVICE> inp(matrix1);
}

TEST(DarkentTensor, CPUActivation)
{
    darknet::tensor::TensorShape shape({9, 15});
    auto matrix1 = std::make_shared<darknet::tensor::Tensor<float, darknet::DeviceType::CPUDEVICE>> (shape);
    std::vector<float> inputs(shape.numElem());
    for(int i = 0; i < shape.numElem(); i++)
        inputs[i] = i / 10;
    matrix1->fromArray(inputs);

    auto inp = std::make_shared<darknet::layer::Input<darknet::DeviceType::CPUDEVICE>>(matrix1);
    inp->init();

    auto inpl = std::static_pointer_cast<darknet::layer::Layer<darknet::DeviceType::CPUDEVICE>>(inp);
    auto act = std::make_shared<darknet::layer::Activation<darknet::DeviceType::CPUDEVICE>>(inpl, darknet::ActivationType::RELU);
    std::shared_ptr<darknet::network::NetworkState> s = nullptr;
    act->forward(s);
}

// TEST(DarkentTensor, GPUActivation)
// {   
//     darknet::tensor::TensorShape shape({9, 15});
//     auto matrix1 = std::make_shared<darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE>> (shape);
//     std::vector<float> inputs(shape.numElem());
//     for(int i = 0; i < shape.numElem(); i++)
//         inputs[i] = i / 10;
//     matrix1->fromArray(inputs);

//     auto inp = std::make_shared<darknet::layer::Input<darknet::DeviceType::GPUDEVICE>>(matrix1);
//     inp->init();
//     auto inpl = std::static_pointer_cast<darknet::layer::Layer<darknet::DeviceType::GPUDEVICE>>(inp);
//     auto act = std::make_shared<darknet::layer::Activation<darknet::DeviceType::GPUDEVICE>>(inpl);
//     std::shared_ptr<darknet::network::NetworkState> s = nullptr;
//     act->forward(s);
// }


int main(int argc, char **argv) {
    // register_handlers();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}