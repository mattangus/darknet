#include "tensor/tensor_gpu.hpp"
#include "utils/template.hpp"

namespace darknet
{
namespace tensor
{
    template<typename T>
    GpuTensor<T>::GpuTensor() : TensorBase<T>(TensorShape({}), DeviceType::GPU), _device(cuda::device::current::get())
    {
        this->data = nullptr;
    }

    template<typename T>
    GpuTensor<T>::GpuTensor(TensorShape& shape) : TensorBase<T>(shape, DeviceType::GPU), _device(cuda::device::current::get())
    {
        this->data = static_cast<T*>(_device.memory().allocate(this->numBytes));
    }

    template<typename T>
    GpuTensor<T>::~GpuTensor()
    {
        if(this->data)
        {
            cuda::memory::device::free(this->data);
        }
    }

    template<typename T>
    std::shared_ptr<TensorBase<T>> GpuTensor<T>::copy()
    {
        auto ret = std::make_shared<GpuTensor>(this->shape);
        cuda::memory::copy(ret->data, this->data, this->numBytes);
        return std::static_pointer_cast<TensorBase<T>>(ret);
    }

    template<typename T>
    std::shared_ptr<TensorBase<T>> GpuTensor<T>::mirror()
    {
        auto ret = std::make_shared<GpuTensor>(this->shape);
        return std::static_pointer_cast<TensorBase<T>>(ret);
    }

    template<typename T>
    std::shared_ptr<TensorBase<T>> GpuTensor<T>::make(TensorShape& shape) {
        auto ret = std::make_shared<GpuTensor<T>>(shape);
        return std::static_pointer_cast<TensorBase<T>>(ret);
    }

    template<typename T>
    void GpuTensor<T>::copyTo(std::shared_ptr<TensorBase<T>>& other)
    {
        assert(other->getShape() == this->shape);
        // let the driver figure out if the memory addresses are cpu or gpu bound.
        cuda::memory::copy(other->ptr(), this->data, this->numBytes);
    }

    template<typename T>
    void GpuTensor<T>::copyTo(std::vector<T>& other)
    {
        if(other.size() != this->numElem)
            other.resize(this->numElem);
        // let the driver figure out if the memory addresses are cpu or gpu bound.
        cuda::memory::copy(&other[0], this->data, this->numBytes);
    }
    template<typename T>
    void GpuTensor<T>::fromArray(std::vector<T>& vec)
    {
        assert(vec.size() == this->numElem);
        cuda::memory::copy(this->data, &vec[0], this->numBytes);
    }

        template<typename T>
    void GpuTensor<T>::operator+=(T other)
    {
        elementwise<gpu::plus<T>>(other);
    }

    template<typename T>
    void GpuTensor<T>::operator+=(const TensorBase<T>& other)
    {
        elementwise<gpu::plus<T>>(other);
    }

    template<typename T>
    void GpuTensor<T>::operator-=(T other)
    {
        elementwise<gpu::minus<T>>(other);
    }
    template<typename T>
    void GpuTensor<T>::operator-=(const TensorBase<T>& other)
    {
        elementwise<gpu::minus<T>>(other);
    }

    template<typename T>
    void GpuTensor<T>::operator*=(T other)
    {
        elementwise<gpu::multiplies<T>>(other);
    }
    template<typename T>
    void GpuTensor<T>::operator*=(const TensorBase<T>& other)
    {
        elementwise<gpu::multiplies<T>>(other);
    }

    template<typename T>
    void GpuTensor<T>::operator/=(T other)
    {
        elementwise<gpu::divides<T>>(other);
    }
    template<typename T>
    void GpuTensor<T>::operator/=(const TensorBase<T>& other)
    {
        elementwise<gpu::divides<T>>(other);
    }

    template<typename T>
    std::shared_ptr<ops::ConvBaseOp<T>> GpuTensor<T>::getConvolution(const std::shared_ptr<TensorBase<T>>& filter, const params::ConvParams& convParams)
    {
        auto op = std::make_shared<ops::gpu::ConvOp<T>>(filter, convParams);
        return std::static_pointer_cast<ops::ConvBaseOp<T>>(op);
    }

    #define GpuTensor(TYPE) template class GpuTensor<TYPE>;

    NUMERIC_TYPES(GpuTensor);

    #undef GpuTensor
    
} // namespace tensor
} // namespace darknet
