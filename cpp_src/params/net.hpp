#pragma once

#include "utils/dict.hpp"
#include "params/layerParams.hpp"
#include "params/strParser.hpp"

namespace darknet
{
namespace params
{

    /**
     * @brief Container for convolution parameters
     * 
     */
    class NetParams : public layerParams {
    public:
        int batchSize;
        int subdivisions;
        int width;
        int height;
        int channels;
        double momentum;
        double decay;
        double angle;
        double saturation;
        double exposure;
        double hue;
        double learningRate;
        int burnIn;
        int maxBatches;
        std::string policy;
        std::vector<int> steps;
        std::vector<double> scales;
        int cutmix;
        int mosaic;

        NetParams()
        {

        }

        static std::shared_ptr<NetParams> parse(std::unordered_map<std::string, std::string>& params)
        {
            std::vector<std::string> required = {"batch", "subdivisions", "width", "height", "learning_rate", "match_batches"};
            darknet::params::StrParamParser strParams(params, required);
            auto ret = std::make_shared<NetParams>();
            ret->batchSize = strParams.get<int>("batch", 1);
            ret->subdivisions = strParams.get<int>("subdivisions", 1);
            ret->width = strParams.get<int>("width", 0);
            ret->height = strParams.get<int>("height", 0);
            ret->channels = strParams.get<int>("channels", 3);
            ret->momentum = strParams.get<double>("momentum", 0.9);
            ret->decay = strParams.get<double>("decay", 0.0001);
            ret->angle = strParams.get<double>("angle", 0);
            ret->saturation = strParams.get<double>("saturation", 1);
            ret->exposure = strParams.get<double>("exposure", 1);
            ret->hue = strParams.get<double>("hue", 0);
            ret->learningRate = strParams.get<double>("learningRate", 0);
            ret->burnIn = strParams.get<int>("burnIn", 0);
            ret->maxBatches = strParams.get<int>("maxBatches", 0);
            ret->policy = strParams.get<std::string>("policy", "constant");
            ret->steps = strParams.getList<int>("steps");
            ret->scales = strParams.getList<double>("scales");
            ret->cutmix = strParams.get<int>("cutmix", 0);
            ret->mosaic = strParams.get<int>("mosaic", 0);
        }
    };

} // namespace params
} // namespace darknet