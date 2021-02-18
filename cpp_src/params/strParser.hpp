#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include "utils/string.hpp"

namespace darknet
{
namespace params
{
    class StrParamParser
    {
    private:
        std::unordered_map<std::string, std::string> params;
        std::vector<std::string> required;
        std::set<std::string> usedParams;

        template<typename T>
        T parse(std::string str)
        {
            T val;
            std::stringstream ss(str);
            ss >> val;
            return val;
        }
    public:
        StrParamParser(std::unordered_map<std::string, std::string>& params, const std::vector<std::string>& required)
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
            T val;
            if(params.count(key) > 0)
                val = parse<T>(params[key]);
            else
                val = defaultVal;
            
            usedParams.emplace(key);
            return val;
        }

        template<typename T>
        T get(std::string key)
        {
            T val;
            if(params.count(key) > 0)
                val = parse<T>(params[key]);
            else
                throw std::runtime_error("Missing key: " + key);
            
            usedParams.emplace(key);
            return val;
        }

        template<typename T>
        std::vector<T> getList(std::string key)
        {
            std::vector<T> ret;
            if(params.count(key) == 0)
                return ret; 
            for(auto v : utils::split(params[key], ','))
            {
                T val = parse<T>(v);
                ret.push_back(val);
            }
            usedParams.emplace(key);
            return ret;
        }

        void warnUnused()
        {
            std::set<std::string> keys;
            for(auto& kv : params)
                keys.emplace(kv.first);
            
            std::set<std::string> result;
            std::set_difference(keys.begin(), keys.end(), usedParams.begin(), usedParams.end(), std::inserter(result, result.begin()));
            for(auto& u : result)
            {
                std::cout << "WARNING! Unused parameter: " << u << std::endl;
            }
        }
    };
} // namespace params
} // namespace darknet
