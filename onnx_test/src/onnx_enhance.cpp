#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/simd_intrinsics.hpp>
#include "onnx_enhance.h"
#include "lut.h"
#include "public.h"


OnnxImgEnhance::OnnxImgEnhance(const std::string onnx_path, size_t num_threads)
            :_onnx_path(onnx_path),
            _env(nullptr),
            _session(nullptr){                                                     
    
    create_onnx_env();
}

OnnxImgEnhance::~OnnxImgEnhance(){
    if(_session != nullptr){
        delete _session;
        
    }

    if(_env != nullptr){
        delete _env;
    }
}

cv::Mat OnnxImgEnhance::run(const cv::Mat &img_rgb, size_t ref_size, std::string lut_cache, bool enable_post){

    cv::Mat img_rgb_normal;
    img_rgb.convertTo(img_rgb_normal, CV_32FC3, 1.0/255.0);

    cv::Size target_size = scale_longe_edge(cv::Size(img_rgb_normal.cols, img_rgb_normal.rows), ref_size);
    cv::Mat in_img;
    cv::resize(img_rgb_normal, in_img, target_size, 0, 0, cv::INTER_AREA);

    cv::Mat nchw_img = cv::dnn::blobFromImage(in_img, 1.0, in_img.size(), cv::Scalar(), false);
    std::cout << "input_img_bgr_normal: "<< img_rgb_normal.rows << "," << img_rgb_normal.cols << "," << img_rgb_normal.channels() << std::endl;
    std::cout << "in_img: " << in_img.rows << "," << in_img.cols << "," << in_img.channels() << std::endl;
    std::cout << "nchw_img: " << nchw_img.rows << "," << nchw_img.cols << "," << nchw_img.channels() << "," << nchw_img.dims << std::endl;

    std::vector<int64_t> target_dims {1,  in_img.channels(), in_img.rows, in_img.cols};
    size_t input_tensor_size = target_dims[0] * target_dims[1] * target_dims[2] * target_dims[3];
    std::vector<float> input_tensor_values(input_tensor_size);
    input_tensor_values.assign(nchw_img.begin<float>(), nchw_img.end<float>());
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names{_session->GetInputName(0, allocator)};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, 
                                                            input_tensor_values.data(), 
                                                            input_tensor_size,
                                                            target_dims.data(),
                                                            target_dims.size()));


    auto output_tensors = _session->Run(Ort::RunOptions{nullptr}, 
                                        input_names.data(), 
                                        input_tensors.data(), 
                                        input_names.size(), 
                                        _output_names.data(), 
                                        _output_names.size());
	// get parameters
    assert(output_tensors.size() == 1);  
    float* f_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    assert(output_shape[1] == _lut_dim);

//    cv::Mat img_enhance_normal = cv::Mat::zeros(img_rgb_normal.size(), img_rgb_normal.type());
//    Lut::trilinear_forward(f_data, img_rgb_normal, img_enhance_normal);
//    cv::Mat img_enhance = img_enhance_normal * 255;
//    img_enhance.convertTo(img_enhance, CV_8UC3);

    int lut_size = 64; // 512 for lut dim = 64, 64 for lut dim = 16
    cv::Mat lut_mat = cv::Mat::zeros(cv::Size(lut_size, lut_size), CV_32FC3);

    std::cout << "start convert lut" << std::endl;
    // 4 for lut dim 16, 8 for lut dim 64
    Lut::convert_lut(f_data, lut_mat, 4, 16);
    if(!lut_cache.empty()){
        cv::Mat ulut = lut_mat * 255;
        ulut.convertTo(ulut, CV_8UC3);
        cv::Mat lut_bgr;
        cv::cvtColor(ulut, lut_bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(lut_cache, lut_bgr);
    }

    std::cout << "start apply lut" << std::endl;

    cv::Mat img_enhance_normal = Lut::trilinear(img_rgb_normal, lut_mat);

    if(enable_post) {
        std::cout << "post lut" << std::endl;
        post_process(img_rgb_normal, img_enhance_normal);
    }

    // cv::Mat img_enhance = triLinear(host_tensor.host<float>(), host_tensor.host<float>(), host_tensor.host<float>(), img_rgb_normal, _lut_dim);
    cv::Mat img_enhance = img_enhance_normal * 255;
    img_enhance.convertTo(img_enhance, CV_8UC3);

    return img_enhance;
}

void OnnxImgEnhance::create_onnx_env(){
    std::cout << "start create onnx env from " << _onnx_path << std::endl;

    _env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "enhance");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    _session = new Ort::Session(*_env, _onnx_path.c_str(), session_options);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    auto cnt = _session->GetOutputCount();
    assert(cnt == 1);

    const char * name = _session->GetOutputName(0, allocator);

    Ort::TypeInfo output_type_info = _session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType output_type = output_tensor_info.GetElementType();
    std::vector<int64_t> output_dims = output_tensor_info.GetShape();

    _lut_channel = output_dims[0];
    _lut_dim = output_dims[1];
    _output_names.push_back(name);
    assert(_lut_dim == 64 || _lut_dim == 16);
    assert(_lut_channel == 3);

    std::cout << "onnx dims = " << _lut_dim << "," << _lut_channel << std::endl;
}