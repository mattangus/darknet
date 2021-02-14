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

            cudnnDataType_t data_type = CUDNN_DATA_FLOAT;

            CHECK_ERROR(cudnnCreateTensorDescriptor(&srcTensorDesc));
            CHECK_ERROR(cudnnCreateTensorDescriptor(&dstTensorDesc));
            CHECK_ERROR(cudnnCreateFilterDescriptor(&weightDesc));
            CHECK_ERROR(cudnnCreateTensorDescriptor(&dsrcTensorDesc));
            CHECK_ERROR(cudnnCreateTensorDescriptor(&ddstTensorDesc));
            CHECK_ERROR(cudnnCreateFilterDescriptor(&dweightDesc));

            CHECK_ERROR(cudnnCreateConvolutionDescriptor(&convDesc));

            // backward delta
            CHECK_ERROR(cudnnSetTensor4dDescriptor(dsrcTensorDesc, CUDNN_TENSOR_NCHW, data_type, this->convParams.batch, this->convParams.c, this->convParams.h, this->convParams.w));
            CHECK_ERROR(cudnnSetTensor4dDescriptor(ddstTensorDesc, CUDNN_TENSOR_NCHW, data_type, this->convParams.batch, this->convParams.out_c, this->convParams.out_h, this->convParams.out_w));
            CHECK_ERROR(cudnnSetFilter4dDescriptor(dweightDesc, data_type, CUDNN_TENSOR_NCHW, this->convParams.n, this->convParams.c / this->convParams.groups, this->convParams.size, this->convParams.size));

            // forward
            CHECK_ERROR(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, data_type, this->convParams.batch, this->convParams.c, this->convParams.h, this->convParams.w));
            CHECK_ERROR(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, data_type, this->convParams.batch, this->convParams.out_c, this->convParams.out_h, this->convParams.out_w));
            CHECK_ERROR(cudnnSetFilter4dDescriptor(weightDesc, data_type, CUDNN_TENSOR_NCHW, this->convParams.n, this->convParams.c / this->convParams.groups, this->convParams.size, this->convParams.size));
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
