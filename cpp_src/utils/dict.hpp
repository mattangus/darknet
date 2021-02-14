#pragma once

namespace darknet
{
namespace utils
{
    /**
     * @brief Helper to get an item or a default value out of a unordered_map
     * 
     * @tparam Key 
     * @tparam Value 
     * @param m 
     * @param key 
     * @param default_value 
     * @return Value& 
     */
    template <typename Key, typename Value>
    Value& get_or(std::unordered_map<Key, Value>& m, const Key& key, Value& default_value)
    {
        auto it = m.find(key);
        if(it == m.end()) {
            return default_value;
        }
        return it->second;
    }
    
} // namespace utils
} // namespace darknet
