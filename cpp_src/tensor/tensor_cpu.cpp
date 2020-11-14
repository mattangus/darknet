#include "tensor/tensor_cpu.hpp"
#include "utils/template.hpp"

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
    void CpuTensor<T>::fromArray(std::vector<T>& vec)
    {
        assert(vec.size() == this->numElem);
        assert(getType<T>() == this->dtype);
        std::memcpy(this->data, &vec[0], this->numBytes);
    }

    #define CPUTENSOR(TYPE) template class CpuTensor<TYPE>;

    NUMERIC_TYPES(CPUTENSOR);

    #undef CPUTENSOR
    
} // namespace tensor
} // namespace darknet
