#pragma once

#include <cmath>
#include <string>
#include <iostream>

#include "layer/layer.hpp"
#include "types/enum.hpp"

namespace darknet
{
namespace layer
{
    class Input : public Layer
    {
    protected:

        std::shared_ptr<tensor::TensorBase<float>> inputTensor;

    public:
        /**
         * @brief Construct a new Activation layer
         * 
         * @param activationType Type of Activation
         */
        Input(std::shared_ptr<tensor::TensorBase<float>>& inputTensor)
            : inputTensor(inputTensor), Layer(nullptr, LayerType::INPUT)
        {
            init();
        }

        void forward() override
        {
            // Do nothing.
        }

        void backward() override
        {
            // do nothing
        }

        void update(int, float, float, float) override
        {
            // do nothing
        }


        void resize() override
        {

        }

        void init()
        {
            this->output = inputTensor;
        }

    };

} // namespace layer
} // namespace darknet