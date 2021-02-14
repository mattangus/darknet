#pragma once

#include "parser/reader.hpp"


namespace darknet
{
namespace parser
{

    class cfgreader : public reader
    {
    private:
        /* data */
    public:
        cfgreader(/* args */) {}
        ~cfgreader() {}

        std::vector<std::string> getLines() override {

        }
    };

} // namespace parser
} // namespace darknet
