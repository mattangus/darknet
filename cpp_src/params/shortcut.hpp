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
    class ShortcutParams : public layerParams {
    public:
        std::string activation;
        std::vector<int> layers;

        ShortcutParams()
        {

        }

        static std::shared_ptr<ShortcutParams> parse(std::unordered_map<std::string, std::string>& params, int layer_num)
        {
            const std::vector<std::string> required = {"from"};
            darknet::params::StrParamParser strParams(params, required);
            auto ret = std::make_shared<ShortcutParams>();
            ret->layers = strParams.getList<int>("from");
            for(int i = 0; i < ret->layers.size(); i++)
            {
                if(ret->layers[i] < 0)
                    ret->layers[i] += layer_num;
            }
            ret->activation = strParams.get<std::string>("activation", "linear");
            
            strParams.warnUnused();
            return ret;
        }
    };

} // namespace params
} // namespace darknet