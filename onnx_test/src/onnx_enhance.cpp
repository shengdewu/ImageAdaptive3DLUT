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

cv::Mat OnnxImgEnhance::run(const cv::Mat &img_rgb, size_t ref_size){

    cv::Mat img_rgb_normal;
    img_rgb.convertTo(img_rgb_normal, CV_32FC3, 1.0/255.0);

    cv::Size target_size = scale_longe_edge(cv::Size(img_rgb_normal.cols, img_rgb_normal.rows), 750);

	cv::Mat in_img;
	cv::resize(img_rgb_normal, in_img, target_size, 0, 0, cv::INTER_AREA);
    cv::Mat nchw_img = cv::dnn::blobFromImage(in_img, 1.0, in_img.size(), cv::Scalar(), false);
    std::cout << "input_img_bgr_normal: "<< img_rgb_normal.rows << "," << img_rgb_normal.cols << "," << img_rgb_normal.channels() << std::endl;
    std::cout << "in_img: " << in_img.rows << "," << in_img.cols << "," << in_img.channels() << std::endl;
    std::cout << "nchw_img: " << nchw_img.rows << "," << nchw_img.cols << "," << nchw_img.channels() << "," << nchw_img.dims << std::endl;

    int d_w = in_img.size().width;
    int d_h = in_img.size().height;
    int d_c = in_img.channels();
    assert(d_c == 3);

    std::vector<int64_t> input_dims {1,  in_img.channels(), in_img.rows, in_img.cols};
    size_t input_tensor_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
    std::vector<float> input_tensor_values(input_tensor_size);
    input_tensor_values.assign(nchw_img.begin<float>(), nchw_img.end<float>());
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names{_session->GetInputName(0, allocator)};
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, 
                                                            input_tensor_values.data(), 
                                                            input_tensor_size, 
                                                            input_dims.data(),
                                                            input_dims.size()));   


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

    int lut_size = 64; // 512 for lut dim = 64, 64 for lut dim = 16
    cv::Mat lut_mat = cv::Mat::zeros(cv::Size(lut_size, lut_size), CV_32FC3);

    std::cout << "start convert lut" << std::endl;
    // 4 for lut dim 16, 8 for lut dim 64
    convert_lut(f_data, lut_mat, 4, 16);
    cv::Mat ulut = lut_mat * 255;
    ulut.convertTo(ulut, CV_8UC3);
    cv::Mat lut_bgr;
    cv::cvtColor(ulut, lut_bgr, cv::COLOR_RGB2BGR);
    cv::imwrite("/mnt/sda1/workspace/ximg/test/base/onnx_lut.jpg", lut_bgr);

    std::cout << "start apply lut" << std::endl;

    cv::Mat img_enhance_normal = Lut::trilinear(img_rgb_normal, lut_mat);

    // cv::Mat img_enhance = triLinear(host_tensor.host<float>(), host_tensor.host<float>(), host_tensor.host<float>(), img_rgb_normal, _lut_dim);
    cv::Mat img_enhance = img_enhance_normal * 255;
    img_enhance.convertTo(img_enhance, CV_8UC3);

    return img_enhance;
}

void OnnxImgEnhance::create_onnx_env(){
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

cv::Size OnnxImgEnhance::scale_longe_edge(cv::Size size, size_t ref_size){
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

void OnnxImgEnhance::convert_lut(const float* lut_prt, cv::Mat &lut_mat, int cell_size, int lut_dim) {
    // cell_size = 4 for lut dim 16, 8 for lut dim 64

    int offset = lut_dim*lut_dim*lut_dim;
    const float* r_ptr = lut_prt;
    const float* g_ptr = lut_prt + offset;
    const float* b_ptr = lut_prt + offset * 2;

    for(int bx=0; bx < cell_size; bx++){
        for(int by=0; by < cell_size; by++){
            for(int g=0; g < lut_dim; g++){
                for(int r=0; r < lut_dim; r++){
                    auto b = bx + by * cell_size;
                    auto x = r + bx * lut_dim;
                    auto y = g + by * lut_dim;

                    int b_offset = lut_dim * lut_dim;
                    int g_offset = lut_dim;

                    lut_mat.at<cv::Vec3f>(y, x)[2] = b_ptr[b * b_offset + g * g_offset + r];
                    lut_mat.at<cv::Vec3f>(y, x)[1] = g_ptr[b * b_offset + g * g_offset + r];
                    lut_mat.at<cv::Vec3f>(y, x)[0] = r_ptr[b * b_offset + g * g_offset + r];
                }
            }
        }
    }
}