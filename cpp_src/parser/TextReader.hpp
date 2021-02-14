#pragma once

#include "parser/reader.hpp"
#include <fstream>


namespace darknet
{
namespace parser
{

    class TextReader : public reader
    {
    private:
        std::string path;
    public:
        TextReader(std::string path)  : path(path) {}
        ~TextReader() {}

        std::vector<std::string> getLines() override {
            std::ifstream file;
            file.open(path);
            if(!file.is_open())
            {
                std::stringstream ss;
                ss << "Could not open config file: " << path;
                throw std::runtime_error(ss.str());
            }
            std::string line;
            std::vector<std::string> ret;
            while (std::getline(file, line))
            {
                ret.push_back(line);
            }
            
            return ret;
        }
    };

} // namespace parser
} // namespace darknet
