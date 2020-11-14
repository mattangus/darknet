
#include <gtest/gtest.h>
#include "tensor/tensor_cpu.hpp"
#include "tensor/tensor_gpu.hpp"


using namespace darknet::tensor;

TEST(DarkentTensor, TestCPUCopy)
{
    TensorShape shape({9, 3});
    
    CpuTensor<float> matrix1;
    ASSERT_EQ(matrix1.ptr(), nullptr);

    auto matrix2 = std::make_shared<CpuTensor<float>>(shape);
    ASSERT_NE(matrix2->ptr(), nullptr);
    auto matrix3 = std::make_shared<CpuTensor<float>>(shape);
    ASSERT_NE(matrix3->ptr(), nullptr);
    auto temp = std::static_pointer_cast<TensorBase<float>>(matrix3);

    matrix2->copyTo(temp);
}

TEST(DarkentTensor, TestGPUCopy)
{
    TensorShape shape({9, 3});
    
    GpuTensor<float> matrix1;
    ASSERT_EQ(matrix1.ptr(), nullptr);

    auto matrix2 = std::make_shared<GpuTensor<float>>(shape);
    ASSERT_NE(matrix2->ptr(), nullptr);
    auto matrix3 = std::make_shared<GpuTensor<float>>(shape);
    ASSERT_NE(matrix3->ptr(), nullptr);
    auto temp = std::static_pointer_cast<TensorBase<float>>(matrix3);

    matrix2->copyTo(temp);
}

TEST(DarkentTensor, TestCPUGPUCopy)
{
    TensorShape shape({9, 3});
    
    auto matrix2 = std::make_shared<CpuTensor<float>>(shape);
    auto matrix3 = std::make_shared<GpuTensor<float>>(shape);
    auto temp = std::static_pointer_cast<TensorBase<float>>(matrix3);

    matrix2->copyTo(temp);
}

TEST(DarkentTensor, TestGPUCPUCopy)
{
    TensorShape shape({9, 3});

    auto matrix2 = std::make_shared<GpuTensor<float>>(shape);
    auto matrix3 = std::make_shared<CpuTensor<float>>(shape);
    auto temp = std::static_pointer_cast<TensorBase<float>>(matrix3);

    matrix2->copyTo(temp);
}

TEST(DarkentTensor, TestGPUFromArray)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<GpuTensor<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);
}

TEST(DarkentTensor, TestCPUFromArray)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<CpuTensor<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);
}


TEST(DarkentTensor, TestGPUFromArray)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<GpuTensor<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);
}

TEST(DarkentTensor, TestCPUFromArray)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<CpuTensor<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);
}


TEST(DarkentTensor, TestGPUPlusEqScaler)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<GpuTensor<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);

    *matrix3 += 1;
}

TEST(DarkentTensor, TestCPUPlusEqScaler)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<CpuTensor<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);
}