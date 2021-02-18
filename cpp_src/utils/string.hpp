#pragma once

#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <sstream>

namespace darknet
{
namespace utils
{

    inline std::string lower(std::string str)
    {
        std::transform(str.begin(), str.end(), str.begin(),
            [](unsigned char c){ return std::tolower(c); });
        
        return str;
    }

    inline std::vector<std::string> split(const std::string &s, char delim) {
        std::stringstream ss(s);
        std::string item;
        std::vector<std::string> elems;
        while (std::getline(ss, item, delim)) {
            elems.push_back(std::move(item));
        }
        return elems;
    }
} // namespace utils
} // namespace darknet

