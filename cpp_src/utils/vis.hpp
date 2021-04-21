#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>

#include "utils/torch.hpp"

namespace darknet
{
namespace vis
{
    int imshow(std::string name, torch::Tensor ten, float scale=1)
    {
        ten = ten - torch::min(ten);
        ten = ten / torch::max(ten);
        auto mat = utils::tensorToMat(ten);
        if(scale != 1)
            cv::resize(mat, mat, cv::Size(), scale, scale);

        cv::imshow(name, mat);
        // return cv::waitKey();
        return 0;
    }
} // namespace vis
} // namespace darknet
