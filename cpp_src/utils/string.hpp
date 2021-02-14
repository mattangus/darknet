#pragma once

#include <string>
#include <algorithm>
#include <cctype>
#include <vector>

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

    inline std::vector<std::string> split(std::string str, std::string delimiter) {
        auto start = 0U;
        auto end = str.find(delimiter);
        std::vector<std::string> result;
        while (end != std::string::npos)
        {
            result.push_back(str.substr(start, end - start));
            start = end + delimiter.length();
            end = str.find(delimiter, start);
        }
        return result;
    }
} // namespace utils
} // namespace darknet

