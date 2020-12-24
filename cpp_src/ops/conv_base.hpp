#pragma once

#include "ops/visitor.hpp"
#include "tensor/tensor_base.hpp"
#include "params/convolution.hpp"
#include "types/enum.hpp"

using namespace darknet::tensor;

namespace darknet
{
namespace ops
{
    
    template<typename T>
    class ConvBaseOp : public OpVisitor<T>
    {
    protected:
        std::shared_ptr<TensorBase<T>> filter;
        std::shared_ptr<TensorBase<T>> input;
        std::shared_ptr<TensorBase<T>> output;
        params::ConvParams convParams;
        DeviceType device;
    public:
        ConvBaseOp(const std::shared_ptr<TensorBase<T>>& input, const std::shared_ptr<TensorBase<T>>& output,
                    const std::shared_ptr<TensorBase<T>>& filter, const params::ConvParams& convParams, DeviceType device)
            : input(input), output(output), filter(filter), convParams(convParams), device(device)
        {
            assert(filter->getDevice() == device);
            assert(input->getDevice() == device);

        }

        ~ConvBaseOp()
        {

        }
        
    };

    // template<typename T>
    // ConvOp<T>::ConvOp(const TensorBase<T>& filter, const params::ConvParams& convParams)
    // {
    // }

    // ConvOp::~ConvOp()
    // {
    // }

} // namespace ops
} // namespace darknet
