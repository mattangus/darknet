#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace darknet
{
namespace utils
{
    torch::Tensor matToTensor(const cv::Mat &image)
    {
        bool isChar = (image.type() & 0xF) < 2;
        std::vector<int64_t> dims = {image.rows, image.cols, image.channels()};
        return torch::from_blob(image.data, dims, isChar ? torch::kChar : torch::kFloat).to(torch::kFloat);
    }

    cv::Mat tensorToMat(torch::Tensor &tensor)
    {
        tensor = tensor.cpu(); //(torch::kCPU);
        auto sizes = tensor.sizes();
        int64_t channels;
        if(tensor.dim() == 2)
            channels = 1;
        else
            channels = sizes[2];
        return cv::Mat(cv::Size(sizes[0], sizes[1]), CV_32FC(channels), tensor.data_ptr());
    }
} // namespace utils
} // namespace darknet
