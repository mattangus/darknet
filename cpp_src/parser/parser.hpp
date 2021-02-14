#pragma once
#include <memory>
#include <map>
#include <vector>

#include "parser/reader.hpp"
#include "types/enum.hpp"
#include "parser/netwrokBuilder.hpp"
#include "params/layerParams.hpp"

namespace darknet
{
namespace parser
{

    class NetworkParser
    {
    protected:
        NetworkParser() {}
        typedef std::vector<std::pair<LayerType, std::shared_ptr<params::layerParams>>> sections_t;

        void build(std::shared_ptr<NetworkBuilder>& builder, sections_t& sections)
        {
            for(auto& s : sections) {
                auto lt = s.first;
                auto params = s.second;

                builder->makeLayer(lt, params);
            }
        }

        virtual sections_t parseSections(std::vector<std::string>& lines) = 0;

    public:
        ~NetworkParser() {}


        void parseConfig(std::shared_ptr<reader>& inputReader,
                            std::shared_ptr<NetworkBuilder>& builder)
        {
            auto lines = inputReader->getLines();
            auto sections = parseSections(lines);
            build(builder, sections);
        }
    };

} // namespace parser
} // namespace darknet
