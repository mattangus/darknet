#pragma once

#include <vector>

namespace darknet
{

    class BoundingBox
    {
    private:
        /* data */
    public:
        float cx, cy, h, w;
        BoundingBox(float cx, float cy, float h, float w) : cx(cx), cy(cy), h(h), w(w) {}
        ~BoundingBox() {}
    };

    class Detection
    {
    private:
        int _class_id;
    public:
        
        std::vector<float> scores;
        BoundingBox bbox;

        Detection(BoundingBox& bbox, std::vector<float>& scores) : bbox(bbox), scores(scores), _class_id(-1) {

        }
        ~Detection() {}

        int class_id()
        {
            if(_class_id == -1)
            {
                _class_id = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
            }
            return _class_id;
        }
    };

} // namespace darknet
