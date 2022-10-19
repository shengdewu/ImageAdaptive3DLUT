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

        void convert_lut(const float* lut_prt, cv::Mat &lut_mat, int cell_size, int lut_dim);

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
