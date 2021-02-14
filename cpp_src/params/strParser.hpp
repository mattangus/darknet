#pragma once

#include <vector>
#include <string>
#include <unordered_map>

namespace darknet
{
namespace params
{
    class StrParamParser
    {
    private:
        std::unordered_map<std::string, std::string> params;
        std::vector<std::string> required;
    public:
        StrParamParser(std::unordered_map<std::string, std::string>& params, std::vector<std::string>& required)
            : params(params), required(required)
        {
            for(auto& r : required)
                if(params.count(r) <= 0)
                    throw std::runtime_error("Missing required parameter: " + r);
        }
        ~StrParamParser() {}

        template<typename T>
        T get(std::string key, T defaultVal)
        {

        }

        template<typename T>
        std::vector<T> getList(std::string key)
        {

        }
    };
} // namespace params
} // namespace darknet
