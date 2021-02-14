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
        NetParams()
        {

        }

        static std::shared_ptr<NetParams> parse(std::unordered_map<std::string, std::string>& params)
        {
            std::vector<std::string> required = {"batch", "subdivisions", "width", "height", "learning_rate", "match_batches"};
            darknet::params::StrParamParser strParams(params, required);

        }
    };

} // namespace params
} // namespace darknet