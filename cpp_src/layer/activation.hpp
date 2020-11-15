#pragma once

#include <cmath>
#include <string>
#include <iostream>

#include "layer/layer.hpp"
#include "types/enum.hpp"
#include "layer/activation_fn.hpp"
#include "tensor/helper.hpp"
#include "errors.hpp"

namespace darknet
{
namespace layer
{
    class Activation : Layer
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
                    tensor::applyElementwise<float, linear<float>>(this->output);
                    break;
                case LOGISTIC:
                    tensor::applyElementwise<float, logistic<float>>(this->output);
                    break;
                case LOGGY:
                    tensor::applyElementwise<float, loggy<float>>(this->output);
                    break;
                case RELU:
                    tensor::applyElementwise<float, relu<float>>(this->output);
                    break;
                case RELU6:
                    tensor::applyElementwise<float, relu6<float>>(this->output);
                    break;
                case ELU:
                    tensor::applyElementwise<float, elu<float>>(this->output);
                    break;
                case SELU:
                    tensor::applyElementwise<float, selu<float>>(this->output);
                    break;
                case GELU:
                    tensor::applyElementwise<float, gelu<float>>(this->output);
                    break;
                case RELIE:
                    tensor::applyElementwise<float, relie<float>>(this->output);
                    break;
                case RAMP:
                    tensor::applyElementwise<float, ramp<float>>(this->output);
                    break;
                case REVLEAKY:
                case LEAKY:
                    tensor::applyElementwise<float, leaky<float>>(this->output);
                    break;
                case TANH:
                    tensor::applyElementwise<float, tanh<float>>(this->output);
                    break;
                case PLSE:
                    tensor::applyElementwise<float, plse<float>>(this->output);
                    break;
                case STAIR:
                    tensor::applyElementwise<float, stair<float>>(this->output);
                    break;
                case HARDTAN:
                    tensor::applyElementwise<float, hardtan<float>>(this->output);
                    break;
                case LHTAN:
                    tensor::applyElementwise<float, lhtan<float>>(this->output);
                    break;
                default:
                    throw NotImplemented("activation " + std::to_string(actType) + " not implemented");
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
        Activation(std::shared_ptr<Layer> inputLayer, ActivationType actType)
            : actType(actType), Layer(inputLayer, LayerType::ACTIVE)
        {
            init();
        }
        
        Activation(std::shared_ptr<Layer> input, std::string& actType)
            : Activation(input, fromString(actType))
        {

        }

        void forward() override
        {
            // TODO: change from copy to a function that assigns
            this->inputLayer->output->copyTo(this->output);
            activateOnOutput();
        }

        void backward() override
        {

        }
        void update(int, float, float, float) override
        {

        }

        void resize() override
        {

        }

        void init()
        {
            this->output = inputLayer->output->mirror();
        }
    };
} // namespace layer
} // namespace darknet