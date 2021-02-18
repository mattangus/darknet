#pragma once

#include <vector>
#include <string>

#include "utils/dict.hpp"
#include "params/layerParams.hpp"
#include "params/strParser.hpp"

namespace darknet
{
namespace params
{

    /**
     * @brief Container for convolution parameters
     * 
     */
    class YoloParams : public layerParams {
    public:
        
        std::vector<int> mask;
        std::vector<float> anchors;
        int classes;
        int num;
        int max_boxes;
        double jitter;
        double ignore_thresh;
        double truth_thresh;
        int random;
        double scale_x_y;
        double iou_thresh;
        double cls_normalizer;
        double iou_normalizer;
        std::string iou_loss;
        std::string nms_kind;
        double beta_nms;
        double max_delta;

        YoloParams()
        {

        }

        static std::shared_ptr<YoloParams> parse(std::unordered_map<std::string, std::string>& params)
        {
            const std::vector<std::string> required = {"mask", "anchors", "classes", "num"};
            darknet::params::StrParamParser strParams(params, required);
            auto ret = std::make_shared<YoloParams>();
            
            ret->mask = strParams.getList<int>("mask");
            ret->anchors = strParams.getList<float>("anchors");
            ret->classes = strParams.get<int>("classes", 20);
            ret->max_boxes = strParams.get<int>("max_boxes", 0);
            ret->num = strParams.get<int>("num", 0);
            ret->jitter = strParams.get<double>("jitter", 0);
            ret->ignore_thresh = strParams.get<double>("ignore_thresh", 0);
            ret->truth_thresh = strParams.get<double>("truth_thresh", 0);
            ret->random = strParams.get<int>("random", 0);
            ret->scale_x_y = strParams.get<double>("scale_x_y", 0);
            ret->iou_thresh = strParams.get<double>("iou_thresh", 0);
            ret->cls_normalizer = strParams.get<double>("cls_normalizer", 0);
            ret->iou_normalizer = strParams.get<double>("iou_normalizer", 0);
            ret->iou_loss = strParams.get<std::string>("iou_loss", "");
            ret->nms_kind = strParams.get<std::string>("nms_kind", "");
            ret->beta_nms = strParams.get<double>("beta_nms", 0);
            ret->max_delta = strParams.get<double>("max_delta", 0);
            
            strParams.warnUnused();
            return ret;
            
        }
    };

} // namespace params
} // namespace darknet