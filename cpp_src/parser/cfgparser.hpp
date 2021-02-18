#pragma once
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <iostream>
#include <regex>

#include "parser/parser.hpp"
#include "parser/netwrokBuilder.hpp"
#include "parser/reader.hpp"
#include "params/layers.hpp"
#include "types/enum.hpp"
#include "utils/string.hpp"
#include "utils/errors.hpp"

namespace darknet
{
namespace parser
{
    class CfgParser : public NetworkParser
    {
        
    protected:

        typedef std::vector<std::pair<LayerType, std::unordered_map<std::string, std::string>>> sections_str_t;

        sections_str_t splitSections(std::vector<std::string>& lines) {
            std::regex headerRe("\\[(\\w+)\\]");
            std::string whiteSpace = " \t\n\r\v\f";
            LayerType prevType;
            sections_str_t ret;
            bool first = true; // don't keep anything above first section [...]
            for(int i = 0; i < lines.size(); i++)
            {
                std::unordered_map<std::string, std::string> vals;
                std::cmatch m;
                while(i < lines.size() && !std::regex_search(lines[i].c_str(), m, headerRe))
                {

                    std::string cur = lines[i];
                    auto wspace = cur.find_first_not_of(whiteSpace);
                    if(wspace != std::string::npos && cur[wspace] != '#' && cur.size() > 0)
                    {
                        // make sure it's not just white space
                        // and make sure it isn't a comment
                        // now remove whitespace
                        cur.erase(std::remove_if(cur.begin(), cur.end(), ::isspace), cur.end());
                        auto split = utils::split(cur, '=');
                        assert(split.size() == 2);
                        vals[split[0]] = split[1];
                    }
                    i++;
                }
                if(first)
                    first = false; // just move on
                else
                {
                    ret.push_back({prevType, vals});
                }

                prevType = layerFromString(m[1].str());
            }
            return ret;
        }

        sections_t parseParams(sections_str_t& sections_str)
        {
            sections_t sections;
            for(int i = 0; i < sections_str.size(); i++)
            {
                auto& sec = sections_str[i];
                int layerNum = i - 1; // first section is always [net]
                LayerType lt = sec.first;
                std::unordered_map<std::string, std::string>& param_map = sec.second;
                std::shared_ptr<params::layerParams> p;
                if(lt == LayerType::NETWORK) {
                    p = std::static_pointer_cast<params::layerParams>(
                                params::NetParams::parse(param_map));
                }
                else if(lt == LayerType::CONVOLUTIONAL){
                    p = std::static_pointer_cast<params::layerParams>(
                                params::ConvParams::parse(param_map));
                }
                else if(lt == LayerType::LOCAL){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::ACTIVE){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::RNN){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::GRU){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::LSTM){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::CONV_LSTM){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::HISTORY){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::CRNN){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::CONNECTED){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::CROP){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::COST){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::REGION){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::YOLO){
                    p = std::static_pointer_cast<params::layerParams>(
                                params::YoloParams::parse(param_map));
                }
                else if(lt == LayerType::GAUSSIAN_YOLO){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::DETECTION){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::SOFTMAX){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::CONTRASTIVE){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::NORMALIZATION){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::BATCHNORM){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::MAXPOOL){
                    p = std::static_pointer_cast<params::layerParams>(
                                params::MaxPoolParams::parse(param_map));
                }
                else if(lt == LayerType::LOCAL_AVGPOOL){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::REORG){   
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::REORG_OLD){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::AVGPOOL){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::ROUTE){
                    p = std::static_pointer_cast<params::layerParams>(
                                params::RouteParams::parse(param_map, layerNum));
                }
                else if(lt == LayerType::UPSAMPLE){
                    p = std::static_pointer_cast<params::layerParams>(
                                params::UpsampleParams::parse(param_map));
                }
                else if(lt == LayerType::SHORTCUT){
                    p = std::static_pointer_cast<params::layerParams>(
                                params::ShortcutParams::parse(param_map, layerNum));
                }
                else if(lt == LayerType::SCALE_CHANNELS){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::SAM){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::DROPOUT){
                    throw darknet::NotImplemented();
                }
                else if(lt == LayerType::EMPTY){
                    throw darknet::NotImplemented();
                }else{
                }
                sections.push_back({lt, p});
            }
            return sections;
        }

        sections_t parseSections(std::vector<std::string>& lines) override {
            auto sections_str = splitSections(lines);
            auto parsed = parseParams(sections_str);
            return parsed;
        }

    public:
        CfgParser()
        {
        }
        ~CfgParser() {}

    };
} // namespace parser
} // namespace darknet
