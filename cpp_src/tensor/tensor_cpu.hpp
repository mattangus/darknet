#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#include <functional>

#if CUDA
    #include <cuda/api_wrappers.hpp>
#endif

#include "layer/activation_fn.hpp"
#include "tensor/tensor_base.hpp"
#include "ops/cpu/convolution.hpp"

namespace darknet
{
namespace tensor
{
    /**
     * @brief CPU tensor
     * 
     */
    template<typename T>
    class CpuTensor: public TensorBase<T>
    {
    private:
        template<typename F>
        inline void elementwise(T other)
        {
            #pragma omp parallel for
            for (size_t i = 0; i < this->numElem; ++i)
                this->data[i] = F()(this->data[i], other);
        }
        template<typename F>
        inline void elementwise(const TensorBase<T>& other)
        {
            // TODO: broadcasting
            assert(this->shape == other.getShape());
            // only support same device operations
            assert(other.getDevice() == DeviceType::CPU);
            auto temp = other.ptr();
            #pragma omp parallel for
            for (size_t i = 0; i < this->numElem; ++i)
                this->data[i] = F()(this->data[i], temp[i]);
        }
    public:
        CpuTensor();
        CpuTensor(TensorShape& shape);
        ~CpuTensor();

        std::shared_ptr<TensorBase<T>> copy() override;
        std::shared_ptr<TensorBase<T>> mirror() override;
        std::shared_ptr<TensorBase<T>> make(TensorShape& shape) override; 
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

        std::shared_ptr<ops::ConvBaseOp<T>> getConvolution(const std::shared_ptr<TensorBase<T>>& filter, const params::ConvParams& convParams) override;

    };

    template<typename T, typename F>
    void applyElementwise(std::shared_ptr<CpuTensor<T>>& input)
    {
        auto ptr = input->ptr();
        #pragma omp parallel for
        for (size_t i = 0; i < input->getShape().numElem(); ++i)
            ptr[i] = F()(ptr[i]);
    }
    
} // namespace tensor
} // namespace darknet
