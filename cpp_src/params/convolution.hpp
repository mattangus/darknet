#pragma once
#include <utility>

namespace darknet
{
namespace params
{

    /**
     * @brief Container for convolution parameters
     * 
     */
    class ConvParams {
    public:
        int filters;
        int kernelSize;
        int groups;
        std::pair<int,int> strides;
        int dilation;
        int padding;
        ConvParams(int filters, int kernelSize, int groups, std::pair<int,int> strides, int dilation, int padding)
            : filters(filters), kernelSize(kernelSize), groups(groups),
              strides(strides), dilation(dilation), padding(padding)
        {

        }
    };

} // namespace params
} // namespace darknet