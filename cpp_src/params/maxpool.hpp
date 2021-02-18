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

        int stride;
        int size;

        MaxPoolParams()
        {

        }

        static std::shared_ptr<MaxPoolParams> parse(std::unordered_map<std::string, std::string>& params)
        {
            const std::vector<std::string> required = {"stride", "size"};
            darknet::params::StrParamParser strParams(params, required);
            auto ret = std::make_shared<MaxPoolParams>();
            ret->stride = strParams.get<int>("stride");
            ret->size = strParams.get<int>("size");
            
            strParams.warnUnused();
            return ret;
        }
    };

} // namespace params
} // namespace darknet