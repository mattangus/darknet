#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>

#if CUDA
    #include <cuda/api_wrappers.hpp>
#endif

#include "layer/activation_fn.hpp"
#include "tensor/tensor_base.hpp"

namespace darknet
{
namespace tensor
{

    /**
     * @brief CPU tensor
     * 
     */
    template<typename T>
    class GpuTensor: public TensorBase<T>
    {
    private:
        cuda::device_t _device;
    public:
        GpuTensor();
        GpuTensor(TensorShape& shape);
        ~GpuTensor();

        std::shared_ptr<TensorBase<T>> copy() override;
        void copyTo(std::shared_ptr<TensorBase<T>>& other) override;
        void copyTo(std::vector<T>& other) override;
        void fromArray(std::vector<T>& vec);

        void operator+=(T other) override;
        void operator+=(const std::shared_ptr<TensorBase<T>>& other) override;

        void operator-=(T other) override;
        void operator-=(std::shared_ptr<TensorBase<T>>& other) override;

        void operator*=(T other) override;
        void operator*=(std::shared_ptr<TensorBase<T>>& other) override;

        void operator/=(T other) override;
        void operator/=(std::shared_ptr<TensorBase<T>>& other) override;

    };
    
} // namespace tensor
} // namespace darknet
