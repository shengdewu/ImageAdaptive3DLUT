#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>


class OnnxImgEnhance{
    public:
        OnnxImgEnhance(const std::string onnx_path, size_t num_threads=4);

        ~OnnxImgEnhance();

        cv::Mat run(const cv::Mat &img_rgb, size_t ref_size=512, std::string lut_cache="", bool enable_post=true);

    private:
        void create_onnx_env();

        void print_input_info(std::string flag);

        void transfor_data(const cv::Mat &mat);

    private:
        int _lut_dim;
        int _lut_channel;
        std::string _lut_name;
        std::string _onnx_path;
        size_t _num_threads;
        Ort::Env *_env;
        Ort::Session *_session;
        std::vector<const char*> _output_names;

};
