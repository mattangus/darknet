#pragma once

#include "ops/visitor.hpp"
#include "tensor/tensor_base.hpp"
#include "params/convolution.hpp"
#include "types/enum.hpp"

namespace darknet
{
namespace ops
{
    
    template<typename T>
    class ConvBaseOp : public OpVisitor<T>
    {
    protected:
        std::shared_ptr<tensor::TensorBase<T>> filter;
        params::ConvParams convParams;
        DeviceType device;
    public:
        ConvBaseOp(const std::shared_ptr<tensor::TensorBase<T>>& filter, const params::ConvParams& convParams, DeviceType device)
            : filter(filter), convParams(convParams), device(device)
        {

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
