#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>

#include "layer/activation_fn.hpp"
#include "tensor/tensor.hpp"

namespace darknet
{
namespace tensor
{

    /**
     * @brief CPU tensor
     * 
     * @tparam T 
     */
    template<typename T>
    class Tensor<T, DeviceType::CPUDEVICE> : public TensorBase<T, DeviceType::CPUDEVICE>
    {
    private:
        
    public:
        Tensor() : TensorBase<T, DeviceType::CPUDEVICE>(TensorShape({}))
        {

        }

        Tensor(TensorShape& shape) : TensorBase<T, DeviceType::CPUDEVICE>(shape)
        {
            this->data = static_cast<T*>(std::malloc(shape.numElem() * sizeof(T)));
        }

        ~Tensor()
        {
            if(this->data)
            {
                std::free(this->data);
            }
        }

        /**
         * @brief Apply an activation function in place to this tensor.
         * @todo might want to move this somwhere else. Shouldn't be encapsulated here.
         * @tparam A  type of activation.
         */
        template<ActivationType A>
        void apply()
        {
            size_t n = this->shape.numElem();
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                switch(A){
                    case LINEAR:
                        this->data[i] = layer::linear(this->data[i]);
                        break;
                    case LOGISTIC:
                        this->data[i] = layer::logistic(this->data[i]);
                        break;
                    case LOGGY:
                        this->data[i] = layer::loggy(this->data[i]);
                        break;
                    case RELU:
                        this->data[i] = layer::relu(this->data[i]);
                        break;
                    case ELU:
                        this->data[i] = layer::elu(this->data[i]);
                        break;
                    case SELU:
                        this->data[i] = layer::selu(this->data[i]);
                        break;
                    case GELU:
                        this->data[i] = layer::gelu(this->data[i]);
                        break;
                    case RELIE:
                        this->data[i] = layer::relie(this->data[i]);
                        break;
                    case RAMP:
                        this->data[i] = layer::ramp(this->data[i]);
                        break;
                    case REVLEAKY:
                    case LEAKY:
                        this->data[i] = layer::leaky(this->data[i]);
                        break;
                    case TANH:
                        this->data[i] = layer::tanh(this->data[i]);
                        break;
                    case PLSE:
                        this->data[i] = layer::plse(this->data[i]);
                        break;
                    case STAIR:
                        this->data[i] = layer::stair(this->data[i]);
                        break;
                    case HARDTAN:
                        this->data[i] = layer::hardtan(this->data[i]);
                        break;
                    case LHTAN:
                        this->data[i] = layer::lhtan(this->data[i]);
                        break;
                }
            }
        }

        Tensor copy()
        {
            Tensor ret(this->shape);
            std::memcpy(ret.data, this->data, this->shape.numElem() * sizeof(T));
            return ret;
        }

        void copyTo(Tensor& other)
        {
            assert(other.shape == this->shape);
            std::memcpy(other.data, this->data, this->shape.numElem() * sizeof(T));
        }

        void fromArray(std::vector<T>& vec)
        {
            size_t n = this->shape.numElem();
            assert(vec.size() == n);
            std::memcpy(this->data, &vec[0], n * sizeof(T));
        }

    };
    
} // namespace tensor
} // namespace darknet
