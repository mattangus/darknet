#pragma once

#include "ops/conv_base.hpp"
#include "tensor/tensor_base.hpp"
#include "params/convolution.hpp"
#include "gpu/dark.hpp"
#include <cudnn.h>

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
        /**
         * @brief Input tensor descriptor
         * 
         */
        cudnnTensorDescriptor_t srcTensorDesc;
        /**
         * @brief Destination tensor descriptor
         * 
         */
        cudnnTensorDescriptor_t dstTensorDesc;
        /**
         * @brief Weights filter descriptor
         * 
         */
        cudnnFilterDescriptor_t weightDesc;
        /**
         * @brief Delta of source tensor descriptor
         * 
         */
        cudnnTensorDescriptor_t dsrcTensorDesc;
        /**
         * @brief Delta of output tensor descriptor
         * 
         */
        cudnnTensorDescriptor_t ddstTensorDesc;
        /**
         * @brief Delta of weights filter descriptor
         * 
         */
        cudnnFilterDescriptor_t dweightDesc;
        /**
         * @brief Conv algorithm descriptor
         * 
         */
        cudnnConvolutionDescriptor_t convDesc;
    public:
        ConvOp(const std::shared_ptr<TensorBase<T>>& input, const std::shared_ptr<TensorBase<T>>& output,
                const std::shared_ptr<TensorBase<T>>& filter, const params::ConvParams& convParams) : ConvBaseOp<T>(input, output, filter, convParams, DeviceType::GPU)
        {
            create_convolutional_cudnn_tensors();
        }
        ~ConvOp()
        {

        }

        void create_convolutional_cudnn_tensors()
        {
            // TODO: gix the CHECK_ERROR() linking error

            cudnnCreateTensorDescriptor(&srcTensorDesc);
            cudnnCreateTensorDescriptor(&dstTensorDesc);
            cudnnCreateFilterDescriptor(&weightDesc);
            cudnnCreateTensorDescriptor(&dsrcTensorDesc);
            cudnnCreateTensorDescriptor(&ddstTensorDesc);
            cudnnCreateFilterDescriptor(&dweightDesc);

            cudnnCreateConvolutionDescriptor(&convDesc);
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
