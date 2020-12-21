#pragma once

#include "ops/conv_base.hpp"
#include "tensor/tensor_base.hpp"
#include "params/convolution.hpp"

using namespace darknet::tensor;

namespace darknet
{
namespace ops
{
namespace gpu
{
    
    template<typename T>
    class ConvOp : public ConvBaseOp<T>
    {
    private:
        /* data */
    public:
        ConvOp(const std::shared_ptr<TensorBase<T>>& input, const std::shared_ptr<TensorBase<T>>& filter, const params::ConvParams& convParams) : ConvBaseOp<T>(input, filter, convParams, DeviceType::GPU)
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

} // namespace gpu
} // namespace ops
} // namespace darknet
