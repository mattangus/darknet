
#ifndef TEST_NAME
#error You must define "TEST_NAME" before including this file
#endif
#ifndef ATENSOR
#error You must define "ATENSOR" before including this file
#endif

#include <gtest/gtest.h>
#include <vector>
#include "tensor/tensor_cpu.hpp"
#include "tensor/tensor_gpu.hpp"
#include "layer/input.hpp"
#include "layer/activation.hpp"
#include "types/enum.hpp"

using namespace darknet;
using namespace darknet::tensor;
using namespace darknet::layer;


TEST(TEST_NAME, Forward)
{
    TensorShape shape({9, 3});
    auto matrix2 = std::make_shared<ATENSOR<float>>(shape);
    auto temp = std::static_pointer_cast<TensorBase<float>>(matrix2);

    auto inputLayer = std::make_shared<Input>(temp);
    std::vector<ActivationType> types = {
        ActivationType::LOGISTIC,
        ActivationType::RELU,
        ActivationType::RELU6,
        ActivationType::RELIE,
        ActivationType::LINEAR,
        ActivationType::RAMP,
        ActivationType::TANH,
        ActivationType::PLSE,
        ActivationType::REVLEAKY,
        ActivationType::LEAKY,
        ActivationType::ELU,
        ActivationType::LOGGY,
        ActivationType::STAIR,
        ActivationType::HARDTAN,
        ActivationType::LHTAN,
        ActivationType::SELU,
        ActivationType::GELU,
        // ActivationType::SWISH,
        // ActivationType::MISH,
        // ActivationType::HARD_MISH
    };
    for(auto act : types)
    {
        auto actLayer = std::make_shared<Activation>(inputLayer, act);
        actLayer->forward();
        // TODO: Actually check output. Might need to do properties based testing (i.e. check that it's monotonic or always positive)
        // would probably have to split into groups of tests or tediously test them individually.
    }
}