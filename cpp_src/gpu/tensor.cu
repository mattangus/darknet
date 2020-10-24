#include "gpu/tensor.cuh"

#include "layer/activation_fn.hpp"
#include "gpu/helpers.hpp"

namespace darknet
{
namespace kernel
{
    /**
     * @brief Apply activation function in place.
     * 
     * @note This is templated to slightly speed up kernels as the compiler can optimize the switch statement.
     * 
     * @tparam T Type of data
     * @tparam A Activation type
     * @param data pointer to data array
     * @param n number of elements in array
     */
    template<typename T, ActivationType A>
    __global__ void applyFunctor(T* data, size_t n)
    {
        CUDA_1D_KERNEL_LOOP(idx, n)
        {        
            switch(A){
                case LINEAR:
                    data[idx] = layer::linear(data[idx]);
                    break;
                case LOGISTIC:
                    data[idx] = layer::logistic(data[idx]);
                    break;
                case LOGGY:
                    data[idx] = layer::loggy(data[idx]);
                    break;
                case RELU:
                    data[idx] = layer::relu(data[idx]);
                    break;
                case ELU:
                    data[idx] = layer::elu(data[idx]);
                    break;
                case SELU:
                    data[idx] = layer::selu(data[idx]);
                    break;
                case GELU:
                    data[idx] = layer::gelu(data[idx]);
                    break;
                case RELIE:
                    data[idx] = layer::relie(data[idx]);
                    break;
                case RAMP:
                    data[idx] = layer::ramp(data[idx]);
                    break;
                case REVLEAKY:
                case LEAKY:
                    data[idx] = layer::leaky(data[idx]);
                    break;
                case TANH:
                    data[idx] = layer::tanh(data[idx]);
                    break;
                case PLSE:
                    data[idx] = layer::plse(data[idx]);
                    break;
                case STAIR:
                    data[idx] = layer::stair(data[idx]);
                    break;
                case HARDTAN:
                    data[idx] = layer::hardtan(data[idx]);
                    break;
                case LHTAN:
                    data[idx] = layer::lhtan(data[idx]);
                    break;
            }
            // data[idx] = f(data[idx]);
        }
    }
} // namespace kernel

namespace gpu
{
    /**
     * @brief Launch the gpu kernel for activations
     * 
     * @tparam T Type of data
     * @tparam A 
     * @param data 
     * @param n 
     */
    template<typename T, ActivationType A>
    void applyFunctor(T* data, size_t n)
    {
        auto grid = get_number_of_blocks(n);
        kernel::applyFunctor<T, A><<<grid, BLOCK>>>(data, n);
        // TODO: for some reason the macro doesn't compile. need to fix that.
        if (cudaPeekAtLastError() != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(cudaPeekAtLastError()));
    }

    // forward declare the expected types
    #define MAKE_TYPES(TYPE, ACT) template void applyFunctor<TYPE, ACT>(TYPE* data, size_t n);

    #define MAKE_ALL(TYPE)  MAKE_TYPES(TYPE, ActivationType::LOGISTIC); \
                            MAKE_TYPES(TYPE, ActivationType::RELU); \
                            MAKE_TYPES(TYPE, ActivationType::RELU6); \
                            MAKE_TYPES(TYPE, ActivationType::RELIE); \
                            MAKE_TYPES(TYPE, ActivationType::LINEAR); \
                            MAKE_TYPES(TYPE, ActivationType::RAMP); \
                            MAKE_TYPES(TYPE, ActivationType::TANH); \
                            MAKE_TYPES(TYPE, ActivationType::PLSE); \
                            MAKE_TYPES(TYPE, ActivationType::REVLEAKY); \
                            MAKE_TYPES(TYPE, ActivationType::LEAKY); \
                            MAKE_TYPES(TYPE, ActivationType::ELU); \
                            MAKE_TYPES(TYPE, ActivationType::LOGGY); \
                            MAKE_TYPES(TYPE, ActivationType::STAIR); \
                            MAKE_TYPES(TYPE, ActivationType::HARDTAN); \
                            MAKE_TYPES(TYPE, ActivationType::LHTAN); \
                            MAKE_TYPES(TYPE, ActivationType::SELU); \
                            MAKE_TYPES(TYPE, ActivationType::GELU); \
                            MAKE_TYPES(TYPE, ActivationType::SWISH); \
                            MAKE_TYPES(TYPE, ActivationType::MISH); \
                            MAKE_TYPES(TYPE, ActivationType::HARD_MISH); \
                            MAKE_TYPES(TYPE, ActivationType::NORM_CHAN); \
                            MAKE_TYPES(TYPE, ActivationType::NORM_CHAN_SOFTMAX); \
                            MAKE_TYPES(TYPE, ActivationType::NORM_CHAN_SOFTMAX_MAXVAL);

    REAL_TYPES(MAKE_ALL);

    #undef MAKE_ALL
    #undef MAKE_TYPES
} // namespace gpu
} // namespace darknet
