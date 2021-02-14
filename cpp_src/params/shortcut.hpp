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
    class ShortcutParams : public layerParams {
    public:
        ShortcutParams()
        {

        }
    };

} // namespace params
} // namespace darknet