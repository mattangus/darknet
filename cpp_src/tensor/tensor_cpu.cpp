#include "tensor/tensor_cpu.hpp"
#include "utils/template.hpp"
#include "errors.hpp"

#include <type_traits>

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
    void CpuTensor<T>::fromArray(std::vector<T>& vec)
    {
        assert(vec.size() == this->numElem);
        assert(getType<T>() == this->dtype);
        std::memcpy(this->data, &vec[0], this->numBytes);
    }

    template<typename T>
    void CpuTensor<T>::operator+=(T other)
    {
        #pragma omp parallel for
        for (size_t i = 0; i < this->numElem; ++i)
            this->data[i] += other;
    }

    template<typename T>
    void CpuTensor<T>::operator+=(const std::shared_ptr<TensorBase<T>>& other)
    {
        // TODO: broadcasting
        assert(this->shape == other->getShape());
        // only support same device operations
        assert(other->getDevice() == DeviceType::CPU);
        auto temp = std::static_pointer_cast<CpuTensor<T>>(other);
        #pragma omp parallel for
        for (size_t i = 0; i < this->numElem; ++i)
            this->data[i] += temp->data[i];
    }

    template<typename T>
    void CpuTensor<T>::operator-=(T other)
    {

    }
    template<typename T>
    void CpuTensor<T>::operator-=(std::shared_ptr<TensorBase<T>>& other)
    {

    }

    template<typename T>
    void CpuTensor<T>::operator*=(T other)
    {

    }
    template<typename T>
    void CpuTensor<T>::operator*=(std::shared_ptr<TensorBase<T>>& other)
    {

    }

    template<typename T>
    void CpuTensor<T>::operator/=(T other)
    {

    }
    template<typename T>
    void CpuTensor<T>::operator/=(std::shared_ptr<TensorBase<T>>& other)
    {

    }

    // Need to remove cpu operators on half, since it is just for storage on cpu.
    // Actual computations should be done on gpu.
    template<> void CpuTensor<half>::operator+=(half other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator+=(const std::shared_ptr<TensorBase<half>>& other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator-=(half other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator-=(std::shared_ptr<TensorBase<half>>& other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator*=(half other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator*=(std::shared_ptr<TensorBase<half>>& other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator/=(half other) { throw NotImplemented(); }
    template<> void CpuTensor<half>::operator/=(std::shared_ptr<TensorBase<half>>& other) { throw NotImplemented(); }


    #define CPUTENSOR(TYPE) template class CpuTensor<TYPE>;

    NUMERIC_TYPES(CPUTENSOR);

    #undef CPUTENSOR
    
} // namespace tensor
} // namespace darknet
