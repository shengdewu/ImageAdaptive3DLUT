#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>
#include <time.h>
#include <chrono>
#include <opencv2/core/simd_intrinsics.hpp>
#include "enhance.h"
#include "lut.h"
#include "public.h"


ImgEnhance::ImgEnhance(const std::string mnn_path, size_t num_threads)
            :_mnn_path(mnn_path),
            _num_threads(num_threads){

    create_mnn_env();
}

ImgEnhance::~ImgEnhance(){
    if(_mnn_interpreter != nullptr){
        _mnn_interpreter->releaseModel();
        if(_mnn_session != nullptr){
            _mnn_interpreter->releaseSession(_mnn_session);
        }
    }
}

cv::Mat ImgEnhance::run(const cv::Mat &img_rgb, size_t ref_size, std::string lut_cache, bool enable_post){

    cv::Mat img_rgb_normal;
    img_rgb.convertTo(img_rgb_normal, CV_32FC3, 1.0/255.0);

    cv::Size target_size = scale_longe_edge(cv::Size(img_rgb_normal.cols, img_rgb_normal.rows), ref_size);

    cv::Mat in_img;
    cv::resize(img_rgb_normal, in_img, target_size, 0, 0, cv::INTER_AREA);
//    int h_offset = ref_size - target_size.height;
//    int w_offset = ref_size - target_size.width;
//    int top = h_offset / 2;
//    int bottom = h_offset - top;
//    int left = w_offset / 2;
//    int right = w_offset - left;
//    cv::Mat in_img;
//    cv::copyMakeBorder(in_img_tmp, in_img, top, bottom, left, right, cv::BorderTypes::BORDER_CONSTANT);

    cv::Mat nchw_img = cv::dnn::blobFromImage(in_img, 1.0, in_img.size(), cv::Scalar(), false);
    std::cout << "input_img_bgr_normal: "<< img_rgb_normal.rows << "," << img_rgb_normal.cols << "," << img_rgb_normal.channels() << std::endl;
    std::cout << "in_img: " << in_img.rows << "," << in_img.cols << "," << in_img.channels() << std::endl;
    std::cout << "nchw_img: " << nchw_img.rows << "," << nchw_img.cols << "," << nchw_img.channels() << "," << nchw_img.dims << std::endl;

    MNN::Session* mnn_session = create_session();

    auto input_tensor = _mnn_interpreter->getSessionInput(mnn_session, nullptr);
    int input_batch = input_tensor->batch();
    int input_channel = input_tensor->channel();
    int input_height = input_tensor->height();
    int input_width = input_tensor->width();

    std::vector<int> target_dims {1,  in_img.channels(), in_img.rows, in_img.cols};
    std::vector<int> input_dims{input_batch, input_channel, input_height, input_width};
    if(input_dims != target_dims){
        _mnn_interpreter->resizeTensor(input_tensor, target_dims);
        _mnn_interpreter->resizeSession(mnn_session);
    }

    input_tensor = _mnn_interpreter->getSessionInput(mnn_session, nullptr);
    auto nchw_tensor = new MNN::Tensor(input_tensor, input_tensor->getDimensionType());
    // memccpy(nchw_tensor->host<float>(), reinterpret_cast<float*>(const_cast<unsigned char*>(nchw_img.data)), 0, 1*d_c*d_h*d_w); // error
    // memccpy(nchw_tensor->buffer().host, nchw_img.data, 0, 1*d_c*d_h*d_w); // error
    memmove(nchw_tensor->host<float>(), nchw_img.data, 1*target_dims[1]*target_dims[2]*target_dims[3] * sizeof(float));
    input_tensor->copyFromHostTensor(nchw_tensor);
    delete nchw_tensor;

    _mnn_interpreter->runSession(mnn_session);
    auto output_tensors = _mnn_interpreter->getSessionOutputAll(mnn_session);

    MNN::Tensor* tensor = output_tensors.at("out_lut");
    MNN::Tensor host_tensor(tensor, tensor->getDimensionType());
    tensor->copyToHostTensor(&host_tensor);

    auto output_shape = host_tensor.shape();
    assert(output_shape[1] == _lut_dim);
    const float *f_data = host_tensor.host<float>();

//    int lut_size = 64; // 512 for lut dim = 64, 64 for lut dim = 16
//    cv::Mat lut_mat = cv::Mat::zeros(cv::Size(lut_size, lut_size), CV_32FC3);
//
//    std::cout << "start convert lut" << std::endl;
//    // 4 for lut dim 16, 8 for lut dim 64
//    Lut::convert_lut(f_data, lut_mat, 4, 16);
//
//    if(!lut_cache.empty()){
//        cv::Mat ulut = lut_mat * 255;
//        ulut.convertTo(ulut, CV_8UC3);
//        cv::Mat lut_bgr;
//        cv::cvtColor(ulut, lut_bgr, cv::COLOR_RGB2BGR);
//        cv::imwrite(lut_cache, lut_bgr);
//    }
//
//    std::cout << "start apply lut" << std::endl;
//
//     cv::Mat img_enhance_normal = Lut::trilinear(img_rgb_normal, lut_mat);
     cv::Mat img_enhance_normal = Lut::trilinear_forward(f_data, img_rgb_normal);

    if(enable_post){
        std::cout << "post lut" << std::endl;
        post_process(img_rgb_normal, img_enhance_normal);
    }

    cv::Mat img_enhance = img_enhance_normal * 255;
    img_enhance.convertTo(img_enhance, CV_8UC3);

    return img_enhance;

}


MNN::Session* ImgEnhance::create_session(){
    // 2 init schedule configt
    MNN::ScheduleConfig schedule_config;
    schedule_config.numThread = _num_threads;
    MNN::BackendConfig backend_config;
    backend_config.precision = MNN::BackendConfig::Precision_High;
    schedule_config.backendConfig = &backend_config;

    //3 create session
    return _mnn_interpreter->createSession(schedule_config);
}

void ImgEnhance::create_mnn_env(){
    std::cout << "start create mnn env from " << _mnn_path << std::endl;
    // 1. init interpreter
    _mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(_mnn_path.c_str()));

    // 2 init schedule configt
    MNN::ScheduleConfig schedule_config;
    schedule_config.numThread = _num_threads;
    MNN::BackendConfig backend_config;
    backend_config.precision = MNN::BackendConfig::Precision_High;
    schedule_config.backendConfig = &backend_config;

    //3 create session
    _mnn_session = _mnn_interpreter->createSession(schedule_config);

    // //4 init input tensor
    auto tmp_input_map = _mnn_interpreter->getSessionInputAll(_mnn_session);
    assert(1 == tmp_input_map.size());

    auto input_tensor = _mnn_interpreter->getSessionInput(_mnn_session, nullptr);
    // // 5. init input dims
    int input_batch = input_tensor->batch();
    int input_channel = input_tensor->channel();
    int input_height = input_tensor->height();
    int input_width = input_tensor->width();
    std::cout << "input_batch: " << input_batch << std::endl;
    std::cout << "input_channel: " << input_channel << std::endl;
    std::cout << "input_height: " << input_height << std::endl;
    std::cout << "input_width: " << input_width << std::endl;
    assert(input_channel == 3);
    int dimension_type = input_tensor->getDimensionType();
    std::cout << "the input tensor has " << input_tensor->size() << std::endl;
    if(dimension_type == MNN::Tensor::CAFFE){
        //NCHW
        // _mnn_interpreter->resizeTensor(input_tensor, {input_batch, input_channel, input_height, input_width});
        // _mnn_interpreter->resizeSession(_mnn_session);
        std::cout << "Dimension Type is CAFFE, NCHW!\n";
    }
    else if(dimension_type == MNN::Tensor::TENSORFLOW){
        //NHWC
        // _mnn_interpreter->resizeTensor(input_tensor, {input_batch, input_height, input_width, input_channel});
        // _mnn_interpreter->resizeSession(_mnn_session);
        std::cout << "Dimension Type is TENSORFLOW, NHWC!\n";
    }
    else if(dimension_type == MNN::Tensor::CAFFE_C4){
        std::cout << "Dimension Type is CAFFE_C4, skip resizeTensor & resizeSession!\n";
    }

    auto tmp_output_map = _mnn_interpreter->getSessionOutputAll(_mnn_session);

    assert(tmp_output_map.size() == 1);
    _lut_name = tmp_output_map.begin()->first;
    auto output_shape = tmp_output_map.begin()->second->shape();
    _lut_dim = output_shape[1];
    _lut_channel = output_shape[0];
    assert (_lut_dim == 64 || _lut_dim == 16);
    assert (_lut_channel == 3);
    std::cout << "lut name: "<< _lut_name << ", output: " << output_shape << ", dim=" << _lut_dim << ",channel="<<_lut_channel << std::endl;
}


void ImgEnhance::print_input_info(std::string flag)
{
    std::cout << "===============" << flag <<" ==============\n";
    std::map<std::string, MNN::Tensor*> input_tensors = _mnn_interpreter->getSessionInputAll(_mnn_session);
    std::cout << "the input tensor has " << input_tensors.size() << std::endl;
    for(auto it=input_tensors.begin(); it!=input_tensors.end(); it++){
        std::cout << "input name " << it->first << ": ";
        it->second->printShape();
        int dimension_type = it->second->getDimensionType();
        if (dimension_type == MNN::Tensor::CAFFE)
            std::cout << "Dimension Type: (CAFFE/PyTorch/ONNX)NCHW" << "\n";
        else if (dimension_type == MNN::Tensor::TENSORFLOW)
            std::cout << "Dimension Type: (TENSORFLOW)NHWC" << "\n";
        else if (dimension_type == MNN::Tensor::CAFFE_C4)
            std::cout << "Dimension Type: (CAFFE_C4)NC4HW4" << "\n";
    }
    std::cout << "=============== ==============\n";
}

void ImgEnhance::transfor_data(const cv::Mat &mat){
    int d_w =mat.size().width;
    int d_h = mat.size().height;
    int d_c = mat.channels();

    auto input_tensor = _mnn_interpreter->getSessionInput(_mnn_session, nullptr);
    int input_batch = input_tensor->batch();
    int input_channel = input_tensor->channel();
    int input_height = input_tensor->height();
    int input_width = input_tensor->width();
    int dimension_type = input_tensor->getDimensionType();
    print_input_info("resize before");

    std::vector<int> target_dims{1, d_c, d_h, d_w};
    std::vector<int> input_dims{input_batch, input_channel, input_height, input_width};
    if(input_dims != target_dims){
        _mnn_interpreter->resizeTensor(input_tensor, target_dims);
        _mnn_interpreter->resizeSession(_mnn_session);
    }

    input_tensor = _mnn_interpreter->getSessionInput(_mnn_session, nullptr);
    auto nhwc_tensor = new MNN::Tensor(input_tensor, input_tensor->getDimensionType());
    memccpy(nhwc_tensor->host<float>(), reinterpret_cast<float*>(const_cast<unsigned char*>(mat.data)), 0, d_c*d_h*d_w);
    input_tensor->copyFromHostTensor(nhwc_tensor);
    input_tensor = _mnn_interpreter->getSessionInput(_mnn_session, nullptr);
    std::cout << input_tensor->host<float>()[0] << "," << input_tensor->host<float>()[d_h*d_w] << std::endl;
    print_input_info("resize after");
}

