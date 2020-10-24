#pragma once

#include <cmath>
#include <string>
#include <iostream>

#include "layer/layer.hpp"
#include "types/enum.hpp"
#include "layer/activation_fn.hpp"

namespace darknet
{
namespace layer
{
    template <DeviceType D>
    class Activation : Layer<D>
    {
    protected:
        /******************************************************
         * Activation functions                               *
         ******************************************************/
        static ActivationType fromString(std::string& s)
        {
            if (s == "logistic") return ActivationType::LOGISTIC;
            if (s == "swish") return ActivationType::SWISH;
            if (s == "mish") return ActivationType::MISH;
            if (s == "hard_mish") return ActivationType::HARD_MISH;
            if (s == "normalize_channels") return ActivationType::NORM_CHAN;
            if (s == "normalize_channels_softmax") return ActivationType::NORM_CHAN_SOFTMAX;
            if (s == "normalize_channels_softmax_maxval") return ActivationType::NORM_CHAN_SOFTMAX_MAXVAL;
            if (s == "loggy") return ActivationType::LOGGY;
            if (s == "relu") return ActivationType::RELU;
            if (s == "relu6") return ActivationType::RELU6;
            if (s == "elu") return ActivationType::ELU;
            if (s == "selu") return ActivationType::SELU;
            if (s == "gelu") return ActivationType::GELU;
            if (s == "relie") return ActivationType::RELIE;
            if (s == "plse") return ActivationType::PLSE;
            if (s == "hardtan") return ActivationType::HARDTAN;
            if (s == "lhtan") return ActivationType::LHTAN;
            if (s == "linear") return ActivationType::LINEAR;
            if (s == "ramp") return ActivationType::RAMP;
            if (s == "revleaky") return ActivationType::REVLEAKY;
            if (s == "leaky") return ActivationType::LEAKY;
            if (s == "tanh") return ActivationType::TANH;
            if (s == "stair") return ActivationType::STAIR;
            std::cerr << "Couldn't find activation function " << s << ", going with ReLU" << std::endl;
            return ActivationType::RELU;
        }

        /******************************************************
         * Array operations                                   *
         ******************************************************/

        /**
         * @brief Run inplace activation on the output
         * 
         */
        void activateOnOutput()
        {
            switch(actType){
                case LINEAR:
                    this->output->apply(linear);
                case LOGISTIC:
                    this->output->apply(logistic);
                case LOGGY:
                    this->output->apply(loggy);
                case RELU:
                    this->output->apply(relu);
                case ELU:
                    this->output->apply(elu);
                case SELU:
                    this->output->apply(selu);
                case GELU:
                    this->output->apply(gelu);
                case RELIE:
                    this->output->apply(relie);
                case RAMP:
                    this->output->apply(ramp);
                case REVLEAKY:
                case LEAKY:
                    this->output->apply(leaky);
                case TANH:
                    this->output->apply(tanh);
                case PLSE:
                    this->output->apply(plse);
                case STAIR:
                    this->output->apply(stair);
                case HARDTAN:
                    this->output->apply(hardtan);
                case LHTAN:
                    this->output->apply(lhtan);
            }
        }

        void init() override
        {
            output.reset(new tensor::Tensor<float, D>(inputLayer->shape));
        }

        ActivationType actType;

    public:
        /**
         * @brief Construct a new Activation layer
         * 
         * @param activationType Type of Activation
         */
        Activation(std::shared_ptr<Layer> inputLayer, ActivationType actType)
            : actType(actType), Layer(inputLayer, LayerType::ACTIVE)
        {
            
        }
        
        Activation(std::shared_ptr<Layer> input, std::string& actType)
            : Activation(input, fromString(actType))
        {

        }

        void forward(std::shared_ptr<network::NetworkState>& netState) override
        {
            inputLayer->output->copyTo(*output);
            activateOnOutput();
        }

        void backward(std::shared_ptr<network::NetworkState>& netState) override
        {

        }
        void update(int, float, float, float) override
        {

        }

    };

} // namespace layer
} // namespace darknet