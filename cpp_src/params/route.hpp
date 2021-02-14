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
    class RouteParams : public layerParams {
    public:
        RouteParams()
        {

        }
    };

} // namespace params
} // namespace darknet