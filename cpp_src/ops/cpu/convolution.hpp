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
        ConvOp(const std::shared_ptr<TensorBase<T>>& input, const std::shared_ptr<TensorBase<T>>& output,
               const std::shared_ptr<TensorBase<T>>& filter, const params::ConvParams& convParams) : ConvBaseOp<T>(input, output, filter, convParams, DeviceType::CPU)
        {
            
        }
        ~ConvOp()
        {

        }

        void operator()() override {
            //NCHW
            // auto outShape = output->getShape();

            // int m = convParams.n / convParams.groups;
            // int k = convParams.kernelSize*convParams.kernelSize*convParams.filters / convParams.groups;
            // int n = outShape[2]*outShape[3];

            // static int u = 0;
            // u++;

            // for(i = 0; i < convParams.batch; ++i)
            // {
            //     for (j = 0; j < convParams.groups; ++j)
            //     {
            //         float *a = l.weights +j*l.nweights / l.groups;
            //         float *b = state.workspace;
            //         float *c = l.output +(i*l.groups + j)*n*m;

                    
            //         //printf(" l.index = %d - FP32 \n", l.index);
            //         float *im = state.input + (i*l.groups + j)*(l.c / l.groups)*l.h*l.w;
            //         if (l.size == 1 && l.stride == 1 && l.dilation == 1) {
            //             b = im;
            //         }
            //         else {
            //             //im2col_cpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);

            //             im2col_cpu_ext(im,   // input
            //                 l.c / l.groups,     // input channels
            //                 l.h, l.w,           // input size (h, w)
            //                 l.size, l.size,     // kernel size (h, w)
            //                 l.pad * l.dilation, l.pad * l.dilation,       // padding (h, w)
            //                 l.stride_y, l.stride_x, // stride (h, w)
            //                 l.dilation, l.dilation, // dilation (h, w)
            //                 b);                 // output

            //         }
            //         gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
            //     }
            // }
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
