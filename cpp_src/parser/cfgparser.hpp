#pragma once
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <iostream>

#include "parser/parser.hpp"
#include "parser/netwrokBuilder.hpp"
#include "parser/reader.hpp"
#include "types/enum.hpp"

namespace darknet
{
namespace parser
{
    class CfgParser : public NetworkParser
    {
        
    protected:

        sections_t parseSections(std::vector<std::string> lines) override {
            for(int i = 0; i < lines.size(); i++)
            {
                
            }
            return sections_t();
        }

    public:
        CfgParser()
        {
        }
        ~CfgParser() {}

    };
} // namespace parser
} // namespace darknet
