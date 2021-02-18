#pragma once

#include "utils/dict.hpp"
#include "params/layerParams.hpp"

namespace darknet
{
namespace params
{

    /**
     * @brief Container for convolution parameters
     * 
     */
    class MaxPoolParams : public layerParams {
    public:

        int stride_x, stride_y;
        int size;
        int padding;

        MaxPoolParams()
        {

        }

        static std::shared_ptr<MaxPoolParams> parse(std::unordered_map<std::string, std::string>& params)
        {
            const std::vector<std::string> required = {};
            darknet::params::StrParamParser strParams(params, required);
            auto ret = std::make_shared<MaxPoolParams>();
            int stride = strParams.get<int>("stride", 1);
            ret->stride_x = strParams.get<int>("stride_x", stride);
            ret->stride_y = strParams.get<int>("stride_y", stride);
            ret->size = strParams.get<int>("size");
            ret->padding = strParams.get<int>("padding", ret->size - 1)/2;

            
            strParams.warnUnused();
            return ret;
        }
    };

} // namespace params
} // namespace darknet