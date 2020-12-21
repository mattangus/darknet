#include "tensor/tensor_cpu.hpp"
#include "utils/template.hpp"
#include "utils/errors.hpp"
#include "linalg/im2col.h"
#include "linalg/gemm.h"

#include <type_traits>

#define SCALAR_LOOP()

namespace darknet
{
namespace tensor
{
    template<typename T>
    CpuTensor<T>::CpuTensor() : TensorBase<T>(TensorShape({}), DeviceType::CPU)
    {

    }

    template<typename T>
    CpuTensor<T>::CpuTensor(TensorShape& shape) : TensorBase<T>(shape, DeviceType::CPU)
    {
        this->data = (T*)std::malloc(this->numBytes);
    }

    template<typename T>
    CpuTensor<T>::~CpuTensor()
    {
        if(this->data)
        {
            std::free(this->data);
        }
    }

    template<typename T>
    std::shared_ptr<TensorBase<T>> CpuTensor<T>::copy()
    {
        auto ret = std::make_shared<CpuTensor>(this->shape);
        std::memcpy(ret->data, this->data, this->numBytes);
        return std::static_pointer_cast<TensorBase<T>>(ret);
    }

    template<typename T>
    std::shared_ptr<TensorBase<T>> CpuTensor<T>::mirror()
    {
        auto ret = std::make_shared<CpuTensor>(this->shape);
        return std::static_pointer_cast<TensorBase<T>>(ret);
    }

    template<typename T>
    void CpuTensor<T>::copyTo(std::shared_ptr<TensorBase<T>>& other)
    {
        assert(other->getShape() == this->shape);
        #if CUDA
            // let the driver figure out if the memory addresses are cpu or gpu bound.
            cuda::memory::copy(other->ptr(), this->data, this->numBytes);
        #else
            //only have cpu tensors
            // std::shared_ptr<CpuTensor> temp = std::static_pointer_cast<CpuTensor>(other);
            std::memcpy(other->ptr(), this->data, this->numBytes);
        #endif
    }

    template<typename T>
    void CpuTensor<T>::copyTo(std::vector<T>& other)
    {
        if(other.size() != this->numElem)
            other.resize(this->numElem);
        #if CUDA
            // let the driver figure out if the memory addresses are cpu or gpu bound.
            cuda::memory::copy(&other[0], this->data, this->numBytes);
        #else
            //only have cpu tensors
            // std::shared_ptr<CpuTensor> temp = std::static_pointer_cast<CpuTensor>(other);
            std::memcpy(&other[0], this->data, this->numBytes);
        #endif
    }

    template<typename T>
    std::shared_ptr<TensorBase<T>> CpuTensor<T>::make(TensorShape& shape) {
        auto ret = std::make_shared<CpuTensor<T>>(shape);
        return std::static_pointer_cast<TensorBase<T>>(ret);
    }

    template<typename T>
    void CpuTensor<T>::fromArray(std::vector<T>& vec)
    {
        assert(vec.size() == this->numElem);
        assert(getType<T>() == this->dtype);
        std::memcpy(this->data, &vec[0], this->numBytes);
    }

    template<typename T>
    void CpuTensor<T>::operator+=(T other)
    {
        elementwise<std::plus<T>>(other);
    }

    template<typename T>
    void CpuTensor<T>::operator+=(const TensorBase<T>& other)
    {
        elementwise<std::plus<T>>(other);
    }

    template<typename T>
    void CpuTensor<T>::operator-=(T other)
    {
        elementwise<std::minus<T>>(other);
    }
    template<typename T>
    void CpuTensor<T>::operator-=(const TensorBase<T>& other)
    {
        elementwise<std::minus<T>>(other);
    }

    template<typename T>
    void CpuTensor<T>::operator*=(T other)
    {
        elementwise<std::multiplies<T>>(other);
    }
    template<typename T>
    void CpuTensor<T>::operator*=(const TensorBase<T>& other)
    {
        elementwise<std::multiplies<T>>(other);
    }

    template<typename T>
    void CpuTensor<T>::operator/=(T other)
    {
        elementwise<std::divides<T>>(other);
    }
    template<typename T>
    void CpuTensor<T>::operator/=(const TensorBase<T>& other)
    {
        elementwise<std::divides<T>>(other);
    }

    // Need to remove cpu operators on half, since it is just for storage on cpu.
    // Actual computations should be done on gpu.
    template<> void CpuTensor<half>::operator+=(half other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator+=(const TensorBase<half>& other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator-=(half other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator-=(const TensorBase<half>& other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator*=(half other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator*=(const TensorBase<half>& other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator/=(half other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator/=(const TensorBase<half>& other) { throw NotImplemented(); }

    // template<typename T>
    // void CpuTensor<T>::convolve(const TensorBase<T>& filter, const params::ConvParams& convParams) {
    //     assert(filter.getDevice() == this->device);

    //     int m = l.n / l.groups;
    //     int k = l.size*l.size*l.c / l.groups;
    //     int n = out_h*out_w;

    //     static int u = 0;
    //     u++;

    //     for(i = 0; i < l.batch; ++i)
    //     {
    //         for (j = 0; j < l.groups; ++j)
    //         {
    //             float *a = l.weights +j*l.nweights / l.groups;
    //             float *b = state.workspace;
    //             float *c = l.output +(i*l.groups + j)*n*m;

                
    //             //printf(" l.index = %d - FP32 \n", l.index);
    //             float *im = state.input + (i*l.groups + j)*(l.c / l.groups)*l.h*l.w;
    //             if (l.size == 1 && l.stride == 1 && l.dilation == 1) {
    //                 b = im;
    //             }
    //             else {
    //                 //im2col_cpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);

    //                 im2col_cpu_ext(im,   // input
    //                     l.c / l.groups,     // input channels
    //                     l.h, l.w,           // input size (h, w)
    //                     l.size, l.size,     // kernel size (h, w)
    //                     l.pad * l.dilation, l.pad * l.dilation,       // padding (h, w)
    //                     l.stride_y, l.stride_x, // stride (h, w)
    //                     l.dilation, l.dilation, // dilation (h, w)
    //                     b);                 // output

    //             }
    //             gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    //         }
    //     }
    // }


    #define CPUTENSOR(TYPE) template class CpuTensor<TYPE>;

    NUMERIC_TYPES(CPUTENSOR);

    #undef CPUTENSOR
    
} // namespace tensor
} // namespace darknet
