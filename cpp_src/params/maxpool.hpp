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
        MaxPoolParams()
        {

        }
    };

} // namespace params
} // namespace darknet