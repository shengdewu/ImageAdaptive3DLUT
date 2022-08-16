#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>



class ImgEnhance{
    public:
        ImgEnhance(const std::string mnn_path, size_t num_threads=4);

        ~ImgEnhance();

        cv::Mat run(const cv::Mat &img_rgb, size_t ref_size=256);

    private:
        void create_mnn_env();

        void print_input_info(std::string flag);

        void transfor_data(const cv::Mat &mat);

        MNN::Session *create_session();
        
        cv::Size scale_longe_edge(cv::Size size, size_t ref_size);

        // cv::Mat triLinear(const cv::Mat &r_lut, const cv::Mat &g_lut, const cv::Mat &b_lut, const cv::Mat &image, const int dim);
        cv::Mat triLinear(const float *r_lut, const float *g_lut, const float *b_lut, const cv::Mat &image, const int dim);

    private:
        int _lut_dim;
        int _lut_channel;
        std::string _lut_name;
        std::string _mnn_path;
        size_t _num_threads;
        std::shared_ptr<MNN::Interpreter> _mnn_interpreter;
        std::shared_ptr<MNN::CV::ImageProcess> _pretreat;
        MNN::Session *_mnn_session = nullptr;

};
