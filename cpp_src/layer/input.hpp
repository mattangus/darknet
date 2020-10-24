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
    template <DeviceType D>
    class Input : public Layer<D>
    {
    protected:

        std::shared_ptr<tensor::Tensor<float, D>> inputTensor;

    public:
        /**
         * @brief Construct a new Activation layer
         * 
         * @param activationType Type of Activation
         */
        Input(std::shared_ptr<tensor::Tensor<float, D>>& inputTensor)
            : inputTensor(inputTensor), Layer<D>(nullptr, LayerType::INPUT)
        {
            
        }

        void forward(std::shared_ptr<network::NetworkState>& netState) override
        {
            // Do nothing.
        }

        void backward(std::shared_ptr<network::NetworkState>& netState) override
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

        void init() override
        {
            this->output = inputTensor;
        }

    };

} // namespace layer
} // namespace darknet