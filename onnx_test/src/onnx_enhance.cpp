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

    auto init_mat_time = std::chrono::system_clock::now();
    cv::Mat img_rgb_normal;
    img_rgb.convertTo(img_rgb_normal, CV_32FC3, 1.0/255.0);

    cv::Size target_size = scale_longe_edge(cv::Size(img_rgb_normal.cols, img_rgb_normal.rows), ref_size);
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
    input_tensor_values.assign(in_img.begin<float>(), in_img.end<float>());
    
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

    cv::Mat lut_mat = cv::Mat::zeros(cv::Size(512, 512), CV_32FC3);    
    int offset = _lut_dim*_lut_dim*_lut_dim;
    const float* r_ptr = f_data;
    const float* g_ptr = f_data + offset;
    const float* b_ptr = f_data + offset * 2;
    
    std::cout << "start convert lut" << std::endl;

    std::ofstream out_file;
    out_file.open("/mnt/sda1/wokspace/ImageAdaptive3DLUT/onnx_test/build/onnx_lut.txt", std::ios::out);
    int rindex = 0, bindex=0, gindex=0;
    for(int bx=0; bx < 8; bx++){
        for(int by=0; by < 8; by++){
            for(int g=0; g < _lut_dim; g++){
                for(int r=0; r < _lut_dim; r++){
                    auto b = bx + by * 8;
                    auto x = r + bx * 64;
                    auto y = g + by * 64;
                    
                    int b_offset = _lut_dim * _lut_dim;
                    int g_offset = _lut_dim;

                    lut_mat.at<cv::Vec3f>(y, x)[2] = b_ptr[b * b_offset + g * g_offset + r];
                    lut_mat.at<cv::Vec3f>(y, x)[1] = g_ptr[b * b_offset + g * g_offset + r]; 
                    lut_mat.at<cv::Vec3f>(y, x)[0] = r_ptr[b * b_offset + g * g_offset + r];

                    // out_file << "lut[" << b << "," << g << "," << r << "]=(" << b_ptr[b * b_offset + g * g_offset + r] << "," << 
                    // g_ptr[b * b_offset + g * g_offset + r] << "," <<  r_ptr[b * b_offset + g * g_offset + r] << ")" << std::endl;

                    rindex = b * b_offset + g * g_offset + r;
                    gindex = b * b_offset + g * g_offset + r + offset;
                    bindex = b * b_offset + g * g_offset + r + offset * 2;
                }
            }
        }
    }

    std::cout << offset * 3 << ":" << rindex << "," << gindex << "," << bindex << std::endl;

    out_file.close();

    cv::Mat ulut = lut_mat * 255;
    ulut.convertTo(ulut, CV_8UC3);
    cv::imwrite("onnx_lut.jpg", ulut);

    std::cout << "start apply lut" << std::endl;

    Lut lut_tool;
    cv::Mat img_enhance_normal = lut_tool.trilinear(img_rgb_normal, lut_mat);

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
    assert(_lut_dim == 64);
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
