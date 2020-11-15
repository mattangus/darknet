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

        /**
         * @brief Apply an activation function in place to this tensor.
         * @todo might want to move this somwhere else. Shouldn't be encapsulated here.
         * @tparam A  type of activation.
         */
        template<ActivationType A>
        void apply()
        {
            // #pragma omp parallel for
            // for (size_t i = 0; i < this->numElem; ++i) {
            //     switch(A){
            //         case LINEAR:
            //             this->data[i] = layer::linear(this->data[i]);
            //             break;
            //         case LOGISTIC:
            //             this->data[i] = layer::logistic(this->data[i]);
            //             break;
            //         case LOGGY:
            //             this->data[i] = layer::loggy(this->data[i]);
            //             break;
            //         case RELU:
            //             this->data[i] = layer::relu(this->data[i]);
            //             break;
            //         case ELU:
            //             this->data[i] = layer::elu(this->data[i]);
            //             break;
            //         case SELU:
            //             this->data[i] = layer::selu(this->data[i]);
            //             break;
            //         case GELU:
            //             this->data[i] = layer::gelu(this->data[i]);
            //             break;
            //         case RELIE:
            //             this->data[i] = layer::relie(this->data[i]);
            //             break;
            //         case RAMP:
            //             this->data[i] = layer::ramp(this->data[i]);
            //             break;
            //         case REVLEAKY:
            //         case LEAKY:
            //             this->data[i] = layer::leaky(this->data[i]);
            //             break;
            //         case TANH:
            //             this->data[i] = layer::tanh(this->data[i]);
            //             break;
            //         case PLSE:
            //             this->data[i] = layer::plse(this->data[i]);
            //             break;
            //         case STAIR:
            //             this->data[i] = layer::stair(this->data[i]);
            //             break;
            //         case HARDTAN:
            //             this->data[i] = layer::hardtan(this->data[i]);
            //             break;
            //         case LHTAN:
            //             this->data[i] = layer::lhtan(this->data[i]);
            //             break;
            //     }
            // }
        }

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
    void applyElementwise(std::shared_ptr<CpuTensor<T>>& input)
    {
        auto ptr = input->ptr();
        #pragma omp parallel for
        for (size_t i = 0; i < input->getShape().numElem(); ++i)
            ptr[i] = F()(ptr[i]);
    }
    
} // namespace tensor
} // namespace darknet
