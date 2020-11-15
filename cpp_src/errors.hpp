#pragma once
#include <stdexcept>
#include <string>

namespace darknet
{
    class NotImplemented : public std::logic_error
    {
    public:
        NotImplemented() : std::logic_error("Function not yet implemented") { };
        NotImplemented(std::string str) : std::logic_error(str) { };
    };
    
} // namespace darknet
