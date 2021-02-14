#pragma once
#include <vector>
#include <string>


namespace darknet
{
namespace parser
{

    class reader
    {
    protected:
        reader() {}
    public:
        
        ~reader() {}
        virtual std::vector<std::string> getLines() = 0;
    };

} // namespace parser
} // namespace darknet
