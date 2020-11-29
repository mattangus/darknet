#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>

#if CUDA
    #include <cuda/api_wrappers.hpp>
#endif

#include "layer/activation_fn.hpp"
#include "tensor/tensor_base.hpp"
#include "gpu/tensor.hpp"

namespace darknet
{
namespace tensor
{
    /**
     * @brief GPU tensor
     * 
     */
    template<typename T>
    class GpuTensor: public TensorBase<T>
    {
    private:
        template<typename F>
        inline void elementwise(T other)
        {
            gpu::elementwise<T, F>(this->data, this->numElem, other);
        }
        template<typename F>
        inline void elementwise(const TensorBase<T>& other)
        {
            // TODO: broadcasting
            assert(this->shape == other.getShape());
            // only support same device operations
            assert(other.getDevice() == DeviceType::GPU);
            gpu::elementwise<T, F>(this->data, this->numElem, other.ptr(), this->numElem);
        }

        cuda::device_t _device;
    public:
        GpuTensor();
        GpuTensor(TensorShape& shape);
        ~GpuTensor();

        std::shared_ptr<TensorBase<T>> copy() override;
        std::shared_ptr<TensorBase<T>> mirror() override;
        void copyTo(std::shared_ptr<TensorBase<T>>& other) override;
        void copyTo(std::vector<T>& other) override;
        void fromArray(std::vector<T>& vec);

        void operator+=(T other) override;
        void operator+=(const TensorBase<T>& other) override;

        void operator-=(T other) override;
        void operator-=(const TensorBase<T>& other) override;

        void operator*=(T other) override;
        void operator*=(const TensorBase<T>& other) override;

        void operator/=(T other) override;
        void operator/=(const TensorBase<T>& other) override;

    };

    template<typename T, typename F>
    void applyElementwise(std::shared_ptr<GpuTensor<T>>& input)
    {
        auto functor = F();
    }
    
} // namespace tensor
} // namespace darknet
