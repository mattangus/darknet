#include "layer/convolutional.hpp"

namespace darknet
{
namespace layer
{
    ConvolutionalLayer::ConvolutionalLayer(std::shared_ptr<Layer> inputLayer, int filters,
            int kernelSize, int groups, std::pair<int, int> strides, int dilation,
            int padding, /* std::shared_ptr<ConvolutionalLayer> share_layer, */
            bool training)
        : Layer(inputLayer, LayerType::CONVOLUTIONAL),
        filters(filters), kernelSize(kernelSize),
        groups(groups), strides(strides),
        dilation(dilation), padding(padding),
        /*share_layer(share_layer),*/ training(training)
    {
        // TODO: No idea why this is needed. this->inputLayer is null without it. this->type is set perfectly fine though????
        this->inputLayer = inputLayer;
        init();
    }

    void ConvolutionalLayer::forward()
    {

    }
    void ConvolutionalLayer::backward(std::shared_ptr<tensor::TensorBase<float>> delta)
    {

    }
    void ConvolutionalLayer::update(int, float, float, float)
    {

    }
    void ConvolutionalLayer::resize()
    {

    }
    void ConvolutionalLayer::init()
    {
        auto inputShape = this->inputLayer->output->getShape();
        assert(inputShape.rank() == 4);
        int batch = inputShape[0];
        int outx = (inputShape[1] + 2*padding - kernelSize) / strides.first + 1;
        int outy = (inputShape[2] + 2*padding - kernelSize) / strides.second + 1;

        auto outShape = tensor::TensorShape({batch, outx, outy, filters});

        this->output = inputLayer->output->make(outShape);
    }
} // namespace layer
} // namespace darknet
