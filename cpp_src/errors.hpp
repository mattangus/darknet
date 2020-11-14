#pragma once
#include <stdexcept>

namespace darknet
{
    class NotImplemented : public std::logic_error
    {
    public:
        NotImplemented() : std::logic_error("Function not yet implemented") { };
    };
    
} // namespace darknet
