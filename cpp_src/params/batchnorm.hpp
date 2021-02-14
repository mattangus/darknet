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
    class BatchnormParams : public layerParams {
    public:
        BatchnormParams()
        {

        }
    };

} // namespace params
} // namespace darknet