#pragma once

#include <memory>
#include <ostream>
#include <cassert>

#include "types/enum.hpp"
#include "tensor/tensor_shape.hpp"

namespace darknet
{
namespace tensor
{

    /**
     * @brief Base class for multi dimensional arrays. This has data and dimension info.
     * 
     * @tparam T Type of data
     * @tparam device device to store data and execute operations on
     */
    template<typename T, DeviceType D>
    class TensorBase
    {
    protected:
        DataType dtype;
        T* data = nullptr;

        /**
         * @brief Hide the constructor so you can't create just the base class. Use Tensor instead
         * 
         * @param shape shape to create
         */
        TensorBase(const TensorShape& shape) : shape(shape)
        {

        }

    public:
        /**
         * @brief Shape of this tensor
         * 
         */
        TensorShape shape;

        /**
         * @brief Get the Device of this tensor
         * 
         * @return DeviceType the device
         */
        DeviceType getDevice() {return D;}

        /**
         * @brief Get the pointer to the data. Caution: this can be pointing to cpu or gpu memory.
         * 
         * @return T* 
         */
        T* ptr() {return data;}

        // virtual TensorBase<T, D> copy() = 0;

        // virtual void copyTo(TensorBase& other) = 0;

        // template<typename F>
        // virtual void apply(F functor, Tensor<T, D>& t1) = 0;

        // template<typename T1, DeviceType device1>
        // friend std::ostream& operator<< (std::ostream& out, const TensorBase<T1, device1>& obj);
    };
    
    // template<typename T, DeviceType device>
    // std::ostream& operator<< (std::ostream& out, const TensorBase<T, device>& obj)
    // {
    //     out << "<TensorBase " << obj.dtype << " " << obj.shape << ">";
    //     return out;
    // }
} // namespace tensor
} // namespace darknet
