#pragma once

#include <memory>
#include <ostream>
#include <cassert>
#include <vector>

#include "types/enum.hpp"
#include "tensor/tensor_shape.hpp"
#include "utils/memory.hpp"
#include "types/enum.hpp"

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
            this->dtype = darknet::getType<T>();
        }

    public:

        // move constructor
        TensorBase(TensorBase&& other) : TensorBase(other.shape, other.device)
        {
            data = other.data;
            other.data = nullptr;
        }

        TensorBase& operator=(TensorBase&& rhs) noexcept {
            dtype = rhs.dtype;
            device = rhs.device;
            numBytes = rhs.numBytes;
            numElem = rhs.numElem;
            data = rhs.data;
            rhs.data = nullptr;
            shape = rhs.shape;
            return *this;
        }

        DataType getType() const { return dtype; }

        TensorShape getShape() const { return shape; }

        /**
         * @brief Get the Device of this tensor
         * 
         * @return DeviceType the device
         */
        DeviceType getDevice() const {return device;}

        /**
         * @brief Get the pointer to the data. Caution: this can be pointing to cpu or gpu memory.
         * 
         * @return T* 
         */
        T* ptr() const {return data;}

        /**
         * @brief Copy this tensor to a new tensor
         * 
         * @return std::shared_ptr<TensorBase<T>> 
         */
        virtual std::shared_ptr<TensorBase<T>> copy() = 0;
        /**
         * @brief Create a new tensor on the same device with the same shape.
         * 
         * @return std::shared_ptr<TensorBase<T>> 
         */
        virtual std::shared_ptr<TensorBase<T>> mirror() = 0;

        /**
         * @brief Convience method for constructing a new tensor on the same device as this tensor.
         * 
         * @param shape Shape to make
         * @return std::shared_ptr<TensorBase<T>> 
         */
        virtual std::shared_ptr<TensorBase<T>> make(TensorShape& shape) = 0;

        /**
         * @brief Copy to another tensor, either 
         * 
         * @param other 
         */
        virtual void copyTo(std::shared_ptr<TensorBase<T>>& other) = 0;
        /**
         * @brief Copy to cpu, storing in a vector.
         * 
         * @param other output vector
         */
        virtual void copyTo(std::vector<T>& other) = 0;

        virtual void operator+=(T other) = 0;
        virtual void operator+=(const TensorBase<T>& other) = 0;

        virtual void operator-=(T other) = 0;
        virtual void operator-=(const TensorBase<T>& other) = 0;

        virtual void operator*=(T other) = 0;
        virtual void operator*=(const TensorBase<T>& other) = 0;

        virtual void operator/=(T other) = 0;
        virtual void operator/=(const TensorBase<T>& other) = 0;

        friend std::ostream& operator<< (std::ostream& out, const TensorBase& obj)
        {
            out << "<TensorBase " << obj.device << " " << obj.dtype << " " << obj.shape << ">";
            return out;
        }
    };
    
} // namespace tensor
} // namespace darknet
