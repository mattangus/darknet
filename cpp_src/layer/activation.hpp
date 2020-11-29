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
    class Activation : public Layer
    {
    protected:
        /******************************************************
         * Activation functions                               *
         ******************************************************/
        static ActivationType fromString(std::string& s);

        /******************************************************
         * Array operations                                   *
         ******************************************************/

        /**
         * @brief Run inplace activation on the output
         * 
         */
        void activateOnOutput();

        void gradOnOutput();

        ActivationType actType;

    public:
        /**
         * @brief Construct a new Activation layer
         * 
         * @param activationType Type of Activation
         */
        Activation(std::shared_ptr<Layer> inputLayer, ActivationType actType);
        Activation(std::shared_ptr<Layer> input, std::string& actType);

        void forward() override;
        void backward(std::shared_ptr<tensor::TensorBase<float>> delta) override;
        void update(int, float, float, float) override;
        void resize() override;
        void init();
    };
} // namespace layer
} // namespace darknet