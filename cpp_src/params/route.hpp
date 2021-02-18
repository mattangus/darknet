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
    class RouteParams : public layerParams {
    public:
        std::vector<int> layers;
        int groups;
        int gorup_id;

        RouteParams()
        {

        }

        static std::shared_ptr<RouteParams> parse(std::unordered_map<std::string, std::string>& params, int layer_num)
        {
            const std::vector<std::string> required = {"layers"};
            darknet::params::StrParamParser strParams(params, required);
            auto ret = std::make_shared<RouteParams>();
            ret->layers = strParams.getList<int>("layers");
            for(int i = 0; i < ret->layers.size(); i++)
            {
                if(ret->layers[i] < 0)
                    ret->layers[i] += layer_num;
            }
            ret->groups = strParams.get<int>("groups", 1);
            ret->gorup_id = strParams.get<int>("gorup_id", 0);
            
            strParams.warnUnused();
            return ret;
        }
    };

} // namespace params
} // namespace darknet