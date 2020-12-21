#pragma once

#include "ops/conv_base.hpp"
#include "tensor/tensor_base.hpp"
#include "params/convolution.hpp"

using namespace darknet::tensor;

namespace darknet
{
namespace ops
{
namespace cpu
{
    
    template<typename T>
    class ConvOp : public ConvBaseOp<T>
    {
    private:
        /* data */
    public:
        ConvOp(const std::shared_ptr<TensorBase<T>>& filter, const params::ConvParams& convParams) : ConvBaseOp<T>(filter, convParams, DeviceType::CPU)
        {

        }
        ~ConvOp()
        {

        }

        void operator()() override {
            
        }
    };

    // template<typename T>
    // ConvOp<T>::ConvOp(const TensorBase<T>& filter, const params::ConvParams& convParams)
    // {
    // }

    // ConvOp::~ConvOp()
    // {
    // }

} // namespace cpu
} // namespace ops
} // namespace darknet
