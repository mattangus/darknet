#pragma once

#include "ops/conv_base.hpp"
#include "ops/cpu/convolution.hpp"
#include "ops/gpu/convolution.hpp"

namespace darknet
{
namespace ops
{
    template<typename T>
    class factory
    {
    public:
        static std::shared_ptr<ops::ConvBaseOp<T>> getConvolution(const std::shared_ptr<TensorBase<T>>& input, const std::shared_ptr<TensorBase<T>>& output,
                                                                    const std::shared_ptr<TensorBase<T>>& filter, const params::ConvParams& convParams)
        {
            auto device = input->getDevice();
            assert(device == filter->getDevice());
            if(device == DeviceType::CPU)
            {
                auto op = std::make_shared<ops::cpu::ConvOp<T>>(input, output, filter, convParams);
                return std::static_pointer_cast<ops::ConvBaseOp<T>>(op);
            }
            else
            {
                auto op = std::make_shared<ops::gpu::ConvOp<T>>(input, output, filter, convParams);
                return std::static_pointer_cast<ops::ConvBaseOp<T>>(op);
            }
        }
    };

} // namespace ops
} // namespace darknet
