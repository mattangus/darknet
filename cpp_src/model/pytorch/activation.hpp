#pragma once

#include <torch/torch.h>
#include "params/layers.hpp"
#include "model/pytorch/dark_module.hpp"
#include "types/enum.hpp"
#include "utils/errors.hpp"

namespace darknet
{
namespace model
{
namespace pytorch
{
    class ActivationModule : public torch::nn::Module
    {
    private:
        /* data */
    public:
        ActivationModule(/* args */) {}
        ~ActivationModule() {}

        virtual torch::Tensor forward(torch::Tensor x) = 0;
    };

    #define SIMPLE_ACTIVATION(NAME, ACT_FN)                 \
    class NAME : public ActivationModule                    \
    {                                                       \
    public:                                                 \
        NAME() {}                                           \
        ~NAME() {}                                          \
        torch::Tensor forward(torch::Tensor x) override {   \
            return ACT_FN(x);                               \
        }                                                   \
    }

    SIMPLE_ACTIVATION(Logistic, torch::sigmoid);
    SIMPLE_ACTIVATION(Relu, torch::relu);
    #define RELU6(x) torch::hardtanh(x, 0, 6);
    SIMPLE_ACTIVATION(Relu6, RELU6);
    // SIMPLE_ACTIVATION(Relie, NotImplemented());
    SIMPLE_ACTIVATION(Linear, );
    // SIMPLE_ACTIVATION(Ramp, NotImplemented());
    SIMPLE_ACTIVATION(Tanh, torch::tanh);
    // SIMPLE_ACTIVATION(Plse, NotImplemented());
    // SIMPLE_ACTIVATION(Revleaky, NotImplemented());
    SIMPLE_ACTIVATION(Leaky, torch::leaky_relu);
    SIMPLE_ACTIVATION(Elu, torch::elu);
    // SIMPLE_ACTIVATION(Loggy, NotImplemented());
    // SIMPLE_ACTIVATION(Stair, NotImplemented());
    SIMPLE_ACTIVATION(Hardtan, torch::hardtanh);
    // SIMPLE_ACTIVATION(Lhtan, NotImplemented());
    SIMPLE_ACTIVATION(Selu, torch::selu);
    SIMPLE_ACTIVATION(Gelu, torch::gelu);
    // SIMPLE_ACTIVATION(Swish, NotImplemented());
    #define MISH(x) x * torch::tanh(torch::softplus(x));
    SIMPLE_ACTIVATION(Mish, MISH);
    // SIMPLE_ACTIVATION(Hard_mish, NotImplemented());
    // SIMPLE_ACTIVATION(Norm_chan, NotImplemented());
    // SIMPLE_ACTIVATION(Norm_chan_softmax, NotImplemented());
    // SIMPLE_ACTIVATION(Norm_chan_softmax_maxval, NotImplemented());

    #undef MISH
    #undef RELU6
    #undef SIMPLE_ACTIVATION

    std::shared_ptr<ActivationModule> getActivation(ActivationType act)
    {
        if(act == ActivationType::LOGISTIC) return std::static_pointer_cast<ActivationModule>(std::make_shared<Logistic>());
        if(act == ActivationType::RELU) return std::static_pointer_cast<ActivationModule>(std::make_shared<Relu>());
        if(act == ActivationType::RELU6) return std::static_pointer_cast<ActivationModule>(std::make_shared<Relu6>());
        if(act == ActivationType::RELIE) throw NotImplemented();
        if(act == ActivationType::LINEAR) return std::static_pointer_cast<ActivationModule>(std::make_shared<Linear>());
        if(act == ActivationType::RAMP) throw NotImplemented();
        if(act == ActivationType::TANH) return std::static_pointer_cast<ActivationModule>(std::make_shared<Tanh>());
        if(act == ActivationType::PLSE) throw NotImplemented();
        if(act == ActivationType::REVLEAKY) throw NotImplemented();
        if(act == ActivationType::LEAKY) return std::static_pointer_cast<ActivationModule>(std::make_shared<Leaky>());
        if(act == ActivationType::ELU) return std::static_pointer_cast<ActivationModule>(std::make_shared<Elu>());
        if(act == ActivationType::LOGGY) throw NotImplemented();
        if(act == ActivationType::STAIR) throw NotImplemented();
        if(act == ActivationType::HARDTAN) return std::static_pointer_cast<ActivationModule>(std::make_shared<Hardtan>());
        if(act == ActivationType::LHTAN) throw NotImplemented();
        if(act == ActivationType::SELU) return std::static_pointer_cast<ActivationModule>(std::make_shared<Selu>());
        if(act == ActivationType::GELU) return std::static_pointer_cast<ActivationModule>(std::make_shared<Gelu>());
        if(act == ActivationType::SWISH) throw NotImplemented();
        if(act == ActivationType::MISH) return std::static_pointer_cast<ActivationModule>(std::make_shared<Mish>());
        if(act == ActivationType::HARD_MISH) throw NotImplemented();
        if(act == ActivationType::NORM_CHAN) throw NotImplemented();
        if(act == ActivationType::NORM_CHAN_SOFTMAX) throw NotImplemented();
        if(act == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL) throw NotImplemented();

        throw std::runtime_error("Activation not found: " + std::to_string(act));
    }

    torch::Tensor activate(torch::Tensor input, ActivationType act)
    {
        if(act == ActivationType::LOGISTIC) return torch::sigmoid(input);
        if(act == ActivationType::RELU) return torch::relu(input);
        if(act == ActivationType::RELU6) return torch::hardtanh(input, 0, 6);
        if(act == ActivationType::RELIE) throw NotImplemented();
        if(act == ActivationType::LINEAR) return input;
        if(act == ActivationType::RAMP) throw NotImplemented();
        if(act == ActivationType::TANH) return torch::tanh(input);
        if(act == ActivationType::PLSE) throw NotImplemented();
        if(act == ActivationType::REVLEAKY) throw NotImplemented();
        if(act == ActivationType::LEAKY) return torch::leaky_relu(input);
        if(act == ActivationType::ELU) return torch::elu(input);
        if(act == ActivationType::LOGGY) throw NotImplemented();
        if(act == ActivationType::STAIR) throw NotImplemented();
        if(act == ActivationType::HARDTAN) return torch::hardtanh(input);
        if(act == ActivationType::LHTAN) throw NotImplemented();
        if(act == ActivationType::SELU) return torch::selu(input);
        if(act == ActivationType::GELU) return torch::gelu(input);
        if(act == ActivationType::SWISH) throw NotImplemented();
        if(act == ActivationType::MISH) return input * torch::tanh(torch::softplus(input));
        if(act == ActivationType::HARD_MISH) throw NotImplemented();
        if(act == ActivationType::NORM_CHAN) throw NotImplemented();
        if(act == ActivationType::NORM_CHAN_SOFTMAX) throw NotImplemented();
        if(act == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL) throw NotImplemented();

        throw std::runtime_error("Activation not found: " + std::to_string(act));
    }

} // namespace torch
} // namespace model
} // namespace darknet
