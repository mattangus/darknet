#pragma once

#include "tensor/tensor_base.hpp"
#include "tensor/tensor_gpu.hpp"
#include "tensor/tensor_cpu.hpp"
#include "types/enum.hpp"

namespace darknet
{
namespace tensor
{

    /**
     * @brief Apply a functor with 1 input element wise.
     * 
     * @note this needs to be in a separate file to avoid cyclical dependencies between tensor_base and derrived.
     * 
     * @tparam T 
     * @tparam F 
     * @param intput 
     */
    template<typename T, typename F>
    void applyElementwise(std::shared_ptr<TensorBase<T>>& input)
    {
        if(input->getDevice() == DeviceType::CPU)
        {
            auto temp = std::static_pointer_cast<CpuTensor<T>>(input);
            applyElementwise<T, F>(temp);
        }
        else if(input->getDevice() == DeviceType::GPU)
        {
            auto temp = std::static_pointer_cast<GpuTensor<T>>(input);
            applyElementwise<T, F>(temp);
        }
    }
    /**
     * @brief Apply a functor with 2 input element wise.
     * 
     * @note this needs to be in a separate file to avoid cyclical dependencies between tensor_base and derrived.
     * 
     * @tparam T 
     * @tparam F 
     * @param intput 
     */
    template<typename T, typename F>
    void applyElementwise(std::shared_ptr<TensorBase<T>>& input1, std::shared_ptr<TensorBase<T>>& input2)
    {
        assert(input1->getDevice() == input2->getDevice());
        if(input1->getDevice() == DeviceType::CPU)
        {
            auto temp1 = std::static_pointer_cast<CpuTensor<T>>(input1);
            auto temp2 = std::static_pointer_cast<CpuTensor<T>>(input2);
            applyElementwise<T, F>(temp1, temp2);
        }
        else if(input1->getDevice() == DeviceType::GPU)
        {
            auto temp1 = std::static_pointer_cast<GpuTensor<T>>(input1);
            auto temp2 = std::static_pointer_cast<GpuTensor<T>>(input2);
            applyElementwise<T, F>(input1, input2);
        }
    }

} // namespace tensor
} // namespace darknet
