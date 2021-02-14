#pragma once

#include <memory>
#include "parser/netwrokBuilder.hpp"


namespace darknet
{
namespace parser
{

    class torchBuilder : public NetworkBuilder
    {
    private:
        /* data */
    public:
        torchBuilder(/* args */) {}
        ~torchBuilder() {}

        void makeConvolutional(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeDeconvolutional(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeConnected(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeMaxpool(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeLocal_avgpool(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeSoftmax(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeDetection(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeDropout(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeCrop(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeRoute(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeCost(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeNormalization(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeAvgpool(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeLocal(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeShortcut(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeScale_channels(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeSam(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeActive(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeRnn(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeGru(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeLstm(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeConv_lstm(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeHistory(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeCrnn(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeBatchnorm(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeNetwork(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeXnor(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeRegion(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeYolo(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeGaussian_yolo(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeIseg(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeReorg(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeReorg_old(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeUpsample(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeLogxent(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeL2norm(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeEmpty(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeBlank(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeContrastive(std::shared_ptr<params::layerParams>& params) override {

        }
        
        void makeInput(std::shared_ptr<params::layerParams>& params) override {

        }
        
    };
} // namespace parser
} // namespace darknet
