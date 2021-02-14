#pragma once

#include <sstream>
#include <unordered_map>
#include <map>

#include "params/layers.hpp"
#include "types/enum.hpp"

namespace darknet
{
namespace parser
{

    class NetworkBuilder
    {
    protected:
        NetworkBuilder() {}
        typedef void (NetworkBuilder::*MFP)(std::shared_ptr<params::layerParams>& params);
        const std::map<LayerType, MFP> layerMap = {
            {LayerType::CONVOLUTIONAL, &NetworkBuilder::makeConvolutional},
            {LayerType::LOCAL, &NetworkBuilder::makeLocal},
            {LayerType::ACTIVE, &NetworkBuilder::makeActive},
            {LayerType::RNN, &NetworkBuilder::makeRnn},
            {LayerType::GRU, &NetworkBuilder::makeGru},
            {LayerType::LSTM, &NetworkBuilder::makeLstm},
            {LayerType::CONV_LSTM, &NetworkBuilder::makeConv_lstm},
            {LayerType::HISTORY, &NetworkBuilder::makeHistory},
            {LayerType::CRNN, &NetworkBuilder::makeCrnn},
            {LayerType::CONNECTED, &NetworkBuilder::makeConnected},
            {LayerType::CROP, &NetworkBuilder::makeCrop},
            {LayerType::COST, &NetworkBuilder::makeCost},
            {LayerType::REGION, &NetworkBuilder::makeRegion},
            {LayerType::YOLO, &NetworkBuilder::makeYolo},
            {LayerType::GAUSSIAN_YOLO, &NetworkBuilder::makeGaussian_yolo},
            {LayerType::DETECTION, &NetworkBuilder::makeDetection},
            {LayerType::SOFTMAX, &NetworkBuilder::makeSoftmax},
            {LayerType::CONTRASTIVE, &NetworkBuilder::makeContrastive},
            {LayerType::NORMALIZATION, &NetworkBuilder::makeNormalization},
            {LayerType::BATCHNORM, &NetworkBuilder::makeBatchnorm},
            {LayerType::MAXPOOL, &NetworkBuilder::makeMaxpool},
            {LayerType::LOCAL_AVGPOOL, &NetworkBuilder::makeLocal_avgpool},
            {LayerType::REORG, &NetworkBuilder::makeReorg},
            {LayerType::REORG_OLD, &NetworkBuilder::makeReorg_old},
            {LayerType::AVGPOOL, &NetworkBuilder::makeAvgpool},
            {LayerType::ROUTE, &NetworkBuilder::makeRoute},
            {LayerType::UPSAMPLE, &NetworkBuilder::makeUpsample},
            {LayerType::SHORTCUT, &NetworkBuilder::makeShortcut},
            {LayerType::SCALE_CHANNELS, &NetworkBuilder::makeScale_channels},
            {LayerType::SAM, &NetworkBuilder::makeSam},
            {LayerType::DROPOUT, &NetworkBuilder::makeDropout},
            {LayerType::EMPTY, &NetworkBuilder::makeEmpty},
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

