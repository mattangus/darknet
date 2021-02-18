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
    class UpsampleParams : public layerParams {
    public:
        int stride;

        UpsampleParams()
        {

        }

        static std::shared_ptr<UpsampleParams> parse(std::unordered_map<std::string, std::string>& params)
        {
            const std::vector<std::string> required = {"stride"};
            darknet::params::StrParamParser strParams(params, required);
            auto ret = std::make_shared<UpsampleParams>();
            ret->stride = strParams.get<int>("stride");
            
            strParams.warnUnused();
            return ret;
        }
    };

} // namespace params
} // namespace darknet