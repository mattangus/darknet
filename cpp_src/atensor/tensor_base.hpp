#pragma once

#include <memory>
#include <ostream>
#include <cassert>

#include "types/enum.hpp"
#include "tensor/tensor_shape.hpp"
#include "utils/memory.hpp"

namespace darknet
{
namespace tensor
{

    /**
     * @brief Base class for multi dimensional arrays. This has data and dimension info.
     * 
     */
    template<typename T>
    class TensorBase
    {
    protected:
        DataType dtype;
        DeviceType device;
        T* data = nullptr;
        size_t numBytes;
        size_t numElem;

        /**
         * @brief Shape of this tensor
         * 
         */
        TensorShape shape;

        /**
         * @brief Hide the constructor so you can't create just the base class. Use Tensor instead
         * 
         * @param shape shape to create
         */
        TensorBase(const TensorShape& shape, DeviceType device) : shape(shape), device(device)
        {
            numElem = this->shape.numElem();
            numBytes = sizeof(T) * numElem;
        }

    public:

        DataType getType() { return dtype; }

        TensorShape getShape() { return shape; }

        /**
         * @brief Get the Device of this tensor
         * 
         * @return DeviceType the device
         */
        DeviceType getDevice() {return device;}

        /**
         * @brief Get the pointer to the data. Caution: this can be pointing to cpu or gpu memory.
         * 
         * @return T* 
         */
        void* ptr() {return data;}

        virtual std::shared_ptr<TensorBase<T>> copy() = 0;

        virtual void copyTo(std::shared_ptr<TensorBase<T>>& other) = 0;

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
