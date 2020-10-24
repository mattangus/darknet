#pragma once

#include <iostream>
#include <cuda/api_wrappers.hpp>

#include "tensor/tensor.hpp"
#include "gpu/tensor.cuh"
#include "types/enum.hpp"

namespace darknet
{
namespace tensor
{

    template<typename T>
    class Tensor<T, DeviceType::GPUDEVICE> : public TensorBase<T, DeviceType::GPUDEVICE>
    {
    private:
        cuda::device_t _device;
    public:

        Tensor() : _device(cuda::device::current::get())
        {
            this->data = nullptr;
        }

        Tensor(TensorShape& shape) : TensorBase<T, DeviceType::GPUDEVICE>(shape), _device(cuda::device::current::get())
        {
            this->data = static_cast<T*>(_device.memory().allocate(shape.numElem() * sizeof(T)));
            std::cout << "creating: " << this->data << std::endl;
        }

        ~Tensor()
        {
            if(this->data)
            {
                std::cout << "freeing: " << this->data << std::endl;
                cuda::memory::device::free(this->data);
            }
        }

        template<ActivationType A>
        void apply()
        {
            size_t n = this->shape.numElem();
            gpu::applyFunctor<T, A>(this->data, n);
        }

        Tensor copy()
        {
            Tensor ret(this->shape);
            cuda::memory::copy(ret.data, this->data, this->shape.numElem() * sizeof(T));
            return ret;
        }

        void copyTo(Tensor& other)
        {
            assert(other.shape == this->shape);
            cuda::memory::copy(other.data, this->data, this->shape.numElem() * sizeof(T));
        }
    };
    
} // namespace tensor
} // namespace darknet
