//
// Created by shengdewu on 22-10-21.
//

#pragma once

#include <opencv2/imgcodecs.hpp>
#include <iostream>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


inline cv::Size scale_longe_edge(cv::Size size, size_t ref_size){
    int target_width = size.width;
    int target_height = size.height;
    int max_size = std::max(target_width, target_height);
    float scale = ref_size * 1.0 / max_size;
    if(scale < 1.0){
        if(target_width > target_height){
            target_width = ref_size;
            target_height = target_height * scale;
        }
        else{
            target_width = target_width * scale;
            target_height = ref_size;
        }
    }

    return cv::Size (int(target_width+0.5), int(target_height+0.5));
}


inline void post_process(const cv::Mat &img_norm, cv::Mat &img_enhance_normal){
    /*# 对插值后的结果添加如下处理代码：
    img_norm img_enhance_normal [0.0, 1.0]
    lut_gain = np.mean(out_img_norm[::10, ::10, :]) / np.mean(input_img_norm[::10, ::10, :])
    w = 32 * ((input_img_norm - 0.5) ** 6)
    if lut_gain > 1:
    w[input_img_norm > 0.5] = 0
    else:
    w[input_img_norm < 0.5] = 0
    out_img_norm = (1 - w) * out_img_norm + w * input_img_norm
    result = np.clip(out_img_norm * 255, 0, 255).astype(np.uint8)*/

    auto src_mean_vec = cv::mean(img_norm);
    auto out_mean_vec = cv::mean(img_enhance_normal);
    float src_mean = float(src_mean_vec[0] + src_mean_vec[1] + src_mean_vec[2]) / 3.f;
    float out_mean = float(out_mean_vec[0] + out_mean_vec[1] + out_mean_vec[2]) / 3.f;
    float lut_gain = out_mean / src_mean;
    std::cout << "lut_gain " << lut_gain << std::endl;
    img_enhance_normal.forEach<cv::Vec3f>([&](cv::Vec3f &rgb, const int position[]) -> void {
        int y = position[0];
        int x = position[1];
        for (int i = 0; i < 3; ++i) {
            float s = img_norm.at<cv::Vec3f>(y, x)[i];
            float w;
            if ((lut_gain > 1.f && s > 0.5f) || (lut_gain <= 1.f && s < 0.5f)) {
                w = 0.f;
            } else {
                w = std::pow((img_norm.at<cv::Vec3f>(y, x)[i] - 0.5f), 6.f) * 32.f;
            }
            rgb[i] = (1.f - w) * rgb[i] + w * s;
        }
    });
}

