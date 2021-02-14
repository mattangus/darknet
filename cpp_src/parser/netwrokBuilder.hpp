#pragma once

#include <sstream>
#include <unordered_map>

#include "params/layers.hpp"

namespace darknet
{
namespace parser
{

    class NetworkBuilder
    {
    protected:
        NetworkBuilder() {}
        typedef void (NetworkBuilder::*MFP)(std::shared_ptr<params::layerParams>& params);
        const std::unordered_map<LayerType, MFP> layerMap = {
            {LayerType::CONVOLUTIONAL, makeConvolutional},
            {LayerType::LOCAL, makeLocal},
            {LayerType::ACTIVE, makeActive},
            {LayerType::RNN, makeRnn},
            {LayerType::GRU, makeGru},
            {LayerType::LSTM, makeLstm},
            {LayerType::CONV_LSTM, makeConv_lstm},
            {LayerType::HISTORY, makeHistory},
            {LayerType::CRNN, makeCrnn},
            {LayerType::CONNECTED, makeConnected},
            {LayerType::CROP, makeCrop},
            {LayerType::COST, makeCost},
            {LayerType::REGION, makeRegion},
            {LayerType::YOLO, makeYolo},
            {LayerType::GAUSSIAN_YOLO, makeGaussian_yolo},
            {LayerType::DETECTION, makeDetection},
            {LayerType::SOFTMAX, makeSoftmax},
            {LayerType::CONTRASTIVE, makeContrastive},
            {LayerType::NORMALIZATION, makeNormalization},
            {LayerType::BATCHNORM, makeBatchnorm},
            {LayerType::MAXPOOL, makeMaxpool},
            {LayerType::LOCAL_AVGPOOL, makeLocal_avgpool},
            {LayerType::REORG, makeReorg},
            {LayerType::REORG_OLD, makeReorg_old},
            {LayerType::AVGPOOL, makeAvgpool},
            {LayerType::ROUTE, makeRoute},
            {LayerType::UPSAMPLE, makeUpsample},
            {LayerType::SHORTCUT, makeShortcut},
            {LayerType::SCALE_CHANNELS, makeScale_channels},
            {LayerType::SAM, makeSam},
            {LayerType::DROPOUT, makeDropout},
            {LayerType::EMPTY, makeEmpty},
        };
    public:
        ~NetworkBuilder() {}

        void makeLayer(LayerType lt, std::shared_ptr<params::layerParams>& params)
        {
            auto func = layerMap.at(lt);
            (this->*func)(params);
        }

        virtual void makeConvolutional(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeDeconvolutional(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeConnected(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeMaxpool(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeLocal_avgpool(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeSoftmax(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeDetection(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeDropout(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeCrop(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeRoute(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeCost(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeNormalization(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeAvgpool(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeLocal(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeShortcut(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeScale_channels(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeSam(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeActive(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeRnn(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeGru(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeLstm(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeConv_lstm(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeHistory(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeCrnn(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeBatchnorm(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeNetwork(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeXnor(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeRegion(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeYolo(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeGaussian_yolo(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeIseg(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeReorg(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeReorg_old(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeUpsample(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeLogxent(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeL2norm(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeEmpty(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeBlank(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeContrastive(std::shared_ptr<params::layerParams>& params) = 0;
        virtual void makeInput(std::shared_ptr<params::layerParams>& params) = 0;
    };

} // namespace parser
} // namespace darknet

