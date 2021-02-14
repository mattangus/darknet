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

        }

    public:
        CfgParser(std::shared_ptr<NetworkBuilder>& builder)
        {

        }
        ~CfgParser() {}

    };
} // namespace parser
} // namespace darknet
