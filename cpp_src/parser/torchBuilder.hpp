#pragma once

#include <memory>
#include <torch/torch.h>
#include <sstream>
#include "parser/netwrokBuilder.hpp"
#include "model/pytorch/modules.hpp"


namespace darknet
{
namespace parser
{
    using model::pytorch::TorchModel;
    using model::pytorch::DarknetModule;
    
    class torchBuilder : public NetworkBuilder
    {
    private:
        int prev_out_depth;
        TorchModel model;
        std::vector<int> outputDepths;
        int layerNum = 0;

        void addModule(std::shared_ptr<DarknetModule>& mod, std::string name, bool outputLayer = false)
        {
            std::stringstream ss;
            ss << name << "_" << layerNum++;
            std::string s = ss.str();
            model.addModule(mod, s, outputLayer);
        }
    public:
        torchBuilder(int input_depth) : prev_out_depth(input_depth)
        {
            outputDepths.push_back(input_depth);
        }
        ~torchBuilder() {}

        void makeConvolutional(std::shared_ptr<params::layerParams>& _params) override {
            auto params = std::static_pointer_cast<params::ConvParams>(_params);
            auto conv = std::static_pointer_cast<DarknetModule>(std::make_shared<model::pytorch::Conv2d>(params, outputDepths));

            addModule(conv, "conv");
        }
        
        void makeDeconvolutional(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeConnected(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeMaxpool(std::shared_ptr<params::layerParams>& _params) override {
            auto params = std::static_pointer_cast<params::MaxPoolParams>(_params);
            // float pad = params->padding;
            // float dilation = 1;
            // float kern = params->size;
            // float hin = 16;
            // float stride = params->stride_x;
            // float out = ((hin + (2*pad) - (dilation * (kern - 1)) - 1)/stride) + 1;
            auto maxpool = std::static_pointer_cast<DarknetModule>(std::make_shared<model::pytorch::MaxPool>(params, outputDepths));

            addModule(maxpool, "maxpool");
        }
        
        void makeLocal_avgpool(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeSoftmax(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeDetection(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeDropout(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeCrop(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeRoute(std::shared_ptr<params::layerParams>& _params) override {
            auto params = std::static_pointer_cast<params::RouteParams>(_params);
            auto route = std::static_pointer_cast<DarknetModule>(std::make_shared<model::pytorch::Route>(params, outputDepths));

            addModule(route, "route");
        }
        
        void makeCost(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeNormalization(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeAvgpool(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeLocal(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeShortcut(std::shared_ptr<params::layerParams>& _params) override {
            auto params = std::static_pointer_cast<params::ShortcutParams>(_params);
            auto shortcut = std::static_pointer_cast<DarknetModule>(std::make_shared<model::pytorch::Shortcut>(params, outputDepths));

            addModule(shortcut, "shortcut");
        }
        
        void makeScale_channels(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeSam(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeActive(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeRnn(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeGru(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeLstm(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeConv_lstm(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeHistory(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeCrnn(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeBatchnorm(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeNetwork(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeXnor(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeRegion(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeYolo(std::shared_ptr<params::layerParams>& _params) override {
            auto params = std::static_pointer_cast<params::YoloParams>(_params);
            auto yolo = std::static_pointer_cast<DarknetModule>(std::make_shared<model::pytorch::Yolo>(params, outputDepths));

            addModule(yolo, "yolo", true);
        }
        
        void makeGaussian_yolo(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeIseg(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeReorg(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeReorg_old(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeUpsample(std::shared_ptr<params::layerParams>& _params) override {
            auto params = std::static_pointer_cast<params::UpsampleParams>(_params);
            auto upsample = std::static_pointer_cast<DarknetModule>(std::make_shared<model::pytorch::Upsample>(params, outputDepths));

            addModule(upsample, "upsample");
        }
        
        void makeLogxent(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeL2norm(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeEmpty(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeBlank(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeContrastive(std::shared_ptr<params::layerParams>& _params) override {

        }
        
        void makeInput(std::shared_ptr<params::layerParams>& _params) override {

        }

        TorchModel getModel() {
            return model;
        }
        
    };
} // namespace parser
} // namespace darknet
