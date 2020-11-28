
// This file is used to 

#ifndef TEST_NAME
#error You must define "TEST_NAME" before including this file
#endif
#ifndef ATENSOR
#error You must define "ATENSOR" before including this file
#endif
#ifndef BTENSOR
#error You must define "BTENSOR" before including this file
#endif

#include <gtest/gtest.h>
#include "tensor/tensor_cpu.hpp"
#include "tensor/tensor_gpu.hpp"

using namespace darknet::tensor;

TEST(TEST_NAME, Copy)
{
    TensorShape shape({9, 3});
    
    ATENSOR<float> matrix1;
    ASSERT_EQ(matrix1.ptr(), nullptr);

    auto matrix2 = std::make_shared<ATENSOR<float>>(shape);
    ASSERT_NE(matrix2->ptr(), nullptr);
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);
    ASSERT_NE(matrix3->ptr(), nullptr);
    auto temp = std::static_pointer_cast<TensorBase<float>>(matrix3);

    matrix2->copyTo(temp);
}


TEST(TEST_NAME, AtoBCopy)
{
    TensorShape shape({9, 3});
    
    auto matrix2 = std::make_shared<ATENSOR<float>>(shape);
    auto matrix3 = std::make_shared<BTENSOR<float>>(shape);
    auto temp = std::static_pointer_cast<TensorBase<float>>(matrix3);

    matrix2->copyTo(temp);
}

TEST(TEST_NAME, FromArray)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);
}

TEST(TEST_NAME, CopyToArray)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);

    std::vector<float> output;

    matrix3->copyTo(output);
}

TEST(TEST_NAME, Make)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);
    auto matrix2 = matrix3->make(shape);
}

TEST(TEST_NAME, PlusEqScaler)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);

    *matrix3 += 1;

    std::vector<float> output;

    matrix3->copyTo(output);

    ASSERT_EQ(input.size(), output.size());
    for(int i = 0; i < input.size(); i++)
        ASSERT_EQ(input[i] + 1, output[i]);
}

TEST(TEST_NAME, MinusEqScaler)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);

    *matrix3 -= 1;

    std::vector<float> output;

    matrix3->copyTo(output);

    ASSERT_EQ(input.size(), output.size());
    for(int i = 0; i < input.size(); i++)
        ASSERT_EQ(input[i] - 1, output[i]);
}

TEST(TEST_NAME, TimesEqScaler)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);

    *matrix3 *= 2;

    std::vector<float> output;

    matrix3->copyTo(output);

    ASSERT_EQ(input.size(), output.size());
    for(int i = 0; i < input.size(); i++)
        ASSERT_EQ(input[i]*2, output[i]);
}

TEST(TEST_NAME, DivEqScaler)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);

    *matrix3 /= 2;

    std::vector<float> output;

    matrix3->copyTo(output);

    ASSERT_EQ(input.size(), output.size());
    for(int i = 0; i < input.size(); i++)
        ASSERT_EQ(input[i]/2, output[i]);
}

TEST(TEST_NAME, PlusEqTensor)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);
    auto matrix2 = std::make_shared<ATENSOR<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);
    matrix2->fromArray(input);
    
    *matrix3 += *matrix2;

    std::vector<float> output;

    matrix3->copyTo(output);

    ASSERT_EQ(input.size(), output.size());
    for(int i = 0; i < input.size(); i++)
        ASSERT_EQ(input[i]*2, output[i]);
}

TEST(TEST_NAME, MinusEqTensor)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);
    auto matrix2 = std::make_shared<ATENSOR<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);
    matrix2->fromArray(input);
    
    *matrix3 -= *matrix2;

    std::vector<float> output;

    matrix3->copyTo(output);

    ASSERT_EQ(input.size(), output.size());
    for(int i = 0; i < input.size(); i++)
        ASSERT_EQ(0, output[i]);
}

TEST(TEST_NAME, TimesEqTensor)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);
    auto matrix2 = std::make_shared<ATENSOR<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i;
    
    matrix3->fromArray(input);
    matrix2->fromArray(input);
    
    *matrix3 *= *matrix2;

    std::vector<float> output;

    matrix3->copyTo(output);

    ASSERT_EQ(input.size(), output.size());
    for(int i = 0; i < input.size(); i++)
        ASSERT_EQ(input[i]*input[i], output[i]);
}

TEST(TEST_NAME, DivEqTensor)
{
    TensorShape shape({9, 3});
    
    auto matrix3 = std::make_shared<ATENSOR<float>>(shape);
    auto matrix2 = std::make_shared<ATENSOR<float>>(shape);

    std::vector<float> input(shape.numElem());
    for(int i = 0; i < input.size(); i++)
        input[i] = i + 1;
    
    matrix3->fromArray(input);
    matrix2->fromArray(input);
    
    *matrix3 /= *matrix2;

    std::vector<float> output;

    matrix3->copyTo(output);

    ASSERT_EQ(input.size(), output.size());
    for(int i = 0; i < input.size(); i++)
        ASSERT_EQ(1, output[i]);
}

