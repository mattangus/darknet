#pragma once

#include <vector>
#include <ostream>

namespace darknet
{
namespace tensor
{
    class TensorShape
    {
    private:
        std::vector<int> dims;
    public:

        TensorShape(std::initializer_list<int> dims) : dims(dims)
        {

        }
        TensorShape(std::vector<int> dims) : dims(dims)
        {

        }
        ~TensorShape()
        {

        }

        int numElem()
        {
            int ret = 1;
            for(auto d : dims)
                ret *= d;
            return ret;
        }

        bool operator==(const TensorShape& other)
        {
            if(other.dims.size() != dims.size())
                return false;
            for(size_t i = 0; i < dims.size(); i++)
                if(dims[i] != other.dims[i])
                    return false;
            return true;
        }
        // friend std::ostream& operator<< (std::ostream& out, const TensorShape& obj);
    };
    
    // std::ostream& operator<< (std::ostream& out, const TensorShape& obj)
    // {
    //     out << "(";
    //     for(int i = 0; i < obj.dims.size(); i++)
    //     {
    //         out << obj.dims[i];
    //         if(i != obj.dims.size() - 1)
    //             out << ", ";
    //     }
    //     out << ")";
    //     return out;
    // }

} // namespace tensor
} // namespace darknet

