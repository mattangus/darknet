
#include "tensor/tensor.hpp"
#include "tensor/tensor_cpu.hpp"
#include "tensor/tensor_gpu.hpp"
#include "tensor/tensor_shape.hpp"
#include "utils/signal_handler.hpp"

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
    auto matrix3 = matrix1.copy();
    matrix1.copyTo(matrix2);
}

TEST(DarkentTensor, TestCopyGPU)
{
    darknet::tensor::TensorShape shape({4, 3});
    darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE> matrix1(shape);
    darknet::tensor::Tensor<float, darknet::DeviceType::GPUDEVICE> matrix2(shape);
    auto matrix3 = matrix1.copy();
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


int main(int argc, char **argv) {
    // register_handlers();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}