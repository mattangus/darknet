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
            // this->output->template apply<actType>();
            // this->output->apply<actType>();
            switch(actType){
                case LINEAR:
                    this->output->template apply<LINEAR>();
                    break;
                case LOGISTIC:
                    this->output->template apply<LOGISTIC>();
                    break;
                case LOGGY:
                    this->output->template apply<LOGGY>();
                    break;
                case RELU:
                    this->output->template apply<RELU>();
                    break;
                case ELU:
                    this->output->template apply<ELU>();
                    break;
                case SELU:
                    this->output->template apply<SELU>();
                    break;
                case GELU:
                    this->output->template apply<GELU>();
                    break;
                case RELIE:
                    this->output->template apply<RELIE>();
                    break;
                case RAMP:
                    this->output->template apply<RAMP>();
                    break;
                case REVLEAKY:
                case LEAKY:
                    this->output->template apply<LEAKY>();
                    break;
                case TANH:
                    this->output->template apply<TANH>();
                    break;
                case PLSE:
                    this->output->template apply<PLSE>();
                    break;
                case STAIR:
                    this->output->template apply<STAIR>();
                    break;
                case HARDTAN:
                    this->output->template apply<HARDTAN>();
                    break;
                case LHTAN:
                    this->output->template apply<LHTAN>();
                    break;
            }
        }

        ActivationType actType;

    public:
        /**
         * @brief Construct a new Activation layer
         * 
         * @param activationType Type of Activation
         */
        Activation(std::shared_ptr<Layer<D>> inputLayer, ActivationType actType)
            : actType(actType), Layer<D>(inputLayer, LayerType::ACTIVE)
        {
            
        }
        
        Activation(std::shared_ptr<Layer<D>> input, std::string& actType)
            : Activation(input, fromString(actType))
        {

        }

        void forward(std::shared_ptr<network::NetworkState>& netState) override
        {
            // TODO: change copy to 2 input fuction
            this->inputLayer->output->copyTo(*(this->output));
            activateOnOutput();
        }

        void backward(std::shared_ptr<network::NetworkState>& netState) override
        {

        }
        void update(int, float, float, float) override
        {

        }

        void resize() override
        {

        }

        void init() override
        {
            this->output.reset(new tensor::Tensor<float, D>(this->inputLayer->output->shape));
        }


    };

} // namespace layer
} // namespace darknet