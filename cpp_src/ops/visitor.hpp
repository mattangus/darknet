#pragma once

#include "tensor/tensor_base.hpp"

namespace darknet
{
namespace tensor
{
    //Forward declare tensor base, since there is a cyclical dependency here.
    //visitor.hpp includes tensor_base.hpp and visversa
    template<typename T>
    class TensorBase;
} // namespace tensor

namespace ops
{

    template<typename T>
    class OpVisitor
    {
    protected:
        
    public:
        virtual void operator()() = 0;
    };
    
} // namespace ops
} // namespace darknet
