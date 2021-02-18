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
    class ConvParams : public layerParams {
    public:
        bool batch_normalize;
        int filters;
        int kernelSize;
        int groups;
        std::pair<int,int> strides;
        int dilation;
        int padding;
        std::string activation;

        ConvParams() {}

        ConvParams(int filters, int kernelSize, int groups, std::pair<int,int> strides, int dilation, int padding)
            : filters(filters), kernelSize(kernelSize), groups(groups),
              strides(strides), dilation(dilation), padding(padding)
        {

        }

        static std::shared_ptr<ConvParams> parse(std::unordered_map<std::string, std::string>& params)
        {
            const std::vector<std::string> required = {"filters", "size"};
            darknet::params::StrParamParser strParams(params, required);
            auto ret = std::make_shared<ConvParams>();
            ret->batch_normalize = strParams.get<bool>("batch_normalize", 0);
            ret->kernelSize = strParams.get<int>("size", 0);
            ret->groups = strParams.get<int>("groups", 1);
            assert(ret->groups > 0);
            
            int stride_x = strParams.get<int>("stride_x", -1);
            int stride_y = strParams.get<int>("stride_y", -1);
            int stride = strParams.get<int>("stride", -1);
            if(stride_x < 0 && stride_y < 0 && stride < 0)
                throw std::runtime_error("a stride must be specified");
            if(stride_x < 0 && stride_y < 0)
            {
                stride_x = stride;
                stride_y = stride;
            }
            ret->strides = {stride_x, stride_y};
                
            bool pad = strParams.get<bool>("pad", false);
            int padding = strParams.get<int>("padding", 0);
            if(pad) ret->padding = ret->kernelSize / 2;

            ret->filters = strParams.get<int>("filters", 0);
            ret->activation = strParams.get<std::string>("activation", "logistic");
            ret->stopbackward = strParams.get<int>("stopbackward", 0);
            
            strParams.warnUnused();
            return ret;
        }
    };

} // namespace params
} // namespace darknet