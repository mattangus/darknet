#pragma once

#include <vector>

namespace darknet
{

    class BoundingBox
    {
    private:
        /* data */
    public:
        BoundingBox(/* args */) {}
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

            }
            return _class_id;
        }
    };

} // namespace darknet
