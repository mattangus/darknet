#pragma once

#include "layer/layer.hpp"
#include "tensor/tensor_shape.hpp"
#include "params/convolution.hpp"

namespace darknet
{
namespace layer
{
    class ConvolutionalLayer : public Layer
    {
    private:

    params::ConvParams convParams;
    std::shared_ptr<Layer> inputLayer;
    std::shared_ptr<ConvolutionalLayer> share_layer;
    bool training;

    public:
        /**
         * @brief Construct a new Convolutional Layer object
         * 
         * @param inputLayer the input to this layer
         * @param filters number of filters to use, specifies the output channel dimension.
         * @param kernelSize positive integer specifying the height and width of the 2D convolution window.
         * @param groups A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups. 
         * @param strides A tuple of 2 integers, specifying the strides of the convolution along the height and width.Specifying any stride value != 1 is incompatible with specifying any dilation value != 1.
         * @param dilation an integer specifying the dilation rate to use for dilated convolution along both axes. specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
         * @param padding The padding to use
         * @param share_layer Weight sharing layer
         * @param training specify if this layer is training.
         */
        ConvolutionalLayer(std::shared_ptr<Layer> inputLayer, int filters,
            int kernelSize, int groups, std::pair<int, int> strides, int dilation,
            int padding, /*std::shared_ptr<ConvolutionalLayer> share_layer,*/
            bool training);
        
        void forward() override;
        void backward(std::shared_ptr<tensor::TensorBase<float>> delta) override;
        void update(int, float, float, float) override;
        void resize() override;
        void init();

    };
    
} // namespace layer
} // namespace darknet