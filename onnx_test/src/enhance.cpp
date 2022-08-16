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

cv::Mat ImgEnhance::run(const cv::Mat &img_rgb, size_t ref_size){

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

    MNN::Session* mnn_session = create_session();

    auto input_tensor = _mnn_interpreter->getSessionInput(mnn_session, nullptr);
    int input_batch = input_tensor->batch();
    int input_channel = input_tensor->channel();
    int input_height = input_tensor->height();
    int input_width = input_tensor->width();

    std::vector<int> target_dims{1, d_c, d_h, d_w};
    std::vector<int> input_dims{input_batch, input_channel, input_height, input_width};
    if(input_dims != target_dims){
        _mnn_interpreter->resizeTensor(input_tensor, target_dims);
        _mnn_interpreter->resizeSession(mnn_session);
    }

    input_tensor = _mnn_interpreter->getSessionInput(mnn_session, nullptr);
    auto nchw_tensor = new MNN::Tensor(input_tensor, input_tensor->getDimensionType());
    // memccpy(nchw_tensor->host<float>(), reinterpret_cast<float*>(const_cast<unsigned char*>(nchw_img.data)), 0, 1*d_c*d_h*d_w); // error
    // memccpy(nchw_tensor->buffer().host, nchw_img.data, 0, 1*d_c*d_h*d_w); // error
    memmove(nchw_tensor->buffer().host, nchw_img.data, 1*d_c*d_h*d_w * sizeof(float));
    input_tensor->copyFromHostTensor(nchw_tensor);
    delete nchw_tensor;

    _mnn_interpreter->runSession(mnn_session);
    auto output_tensors = _mnn_interpreter->getSessionOutputAll(mnn_session);

    MNN::Tensor* tensor = output_tensors.at("out_lut");
    MNN::Tensor host_tensor(tensor, tensor->getDimensionType());
    tensor->copyToHostTensor(&host_tensor);

    auto output_shape = host_tensor.shape();
    assert(output_shape[1] == _lut_dim);

    cv::Mat lut_mat = cv::Mat::zeros(cv::Size(512, 512), CV_32FC3);
    const float *f_data = host_tensor.host<float>();
    
    int offset = _lut_dim*_lut_dim*_lut_dim;
    const float* r_ptr = f_data;
    const float* g_ptr = f_data + offset;
    const float* b_ptr = f_data + offset * 2;
    
    std::cout << "start convert lut" << std::endl;

    std::ofstream out_file;
    out_file.open("/mnt/sda1/wokspace/ImageAdaptive3DLUT/onnx_test/build/tlut.txt", std::ios::out);
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
    cv::imwrite("mnn_lut.jpg", ulut);

    std::cout << "start apply lut" << std::endl;

    Lut lut_tool;
    cv::Mat img_enhance_normal = lut_tool.trilinear(img_rgb_normal, lut_mat);

    // cv::Mat img_enhance = triLinear(host_tensor.host<float>(), host_tensor.host<float>(), host_tensor.host<float>(), img_rgb_normal, _lut_dim);
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
    assert (_lut_dim == 64);
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


cv::Size ImgEnhance::scale_longe_edge(cv::Size size, size_t ref_size){
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


// cv::Mat ImgEnhance::triLinear(const cv::Mat &r_lut, const cv::Mat &g_lut, const cv::Mat &b_lut, const cv::Mat &image, const int dim)
cv::Mat ImgEnhance::triLinear(const float *r_lut, const float *g_lut, const float *b_lut, const cv::Mat &image, const int dim)
{
    cv::Mat img_enhance = cv::Mat::zeros(image.size(), image.type()); //img_rgb_normal.clone();
    
    auto binsize = 1.000001 / (dim - 1);

    image.forEach<cv::Vec3f>([&](cv::Vec3f &rgb, const int position[]) -> void {
		const float r = rgb[0];
		const float g = rgb[1];
		const float b = rgb[2];
        
        int r_id = floor(r / binsize);
        int g_id = floor(g / binsize);
        int b_id = floor(b / binsize);

        float r_d = fmod(r,binsize) / binsize;
        float g_d = fmod(g,binsize) / binsize;
        float b_d = fmod(b,binsize) / binsize;

        int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
        int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
        int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
        int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
        int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
        int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;


        float w000 = (1-r_d)*(1-g_d)*(1-b_d);
        float w100 = r_d*(1-g_d)*(1-b_d);
        float w010 = (1-r_d)*g_d*(1-b_d);
        float w110 = r_d*g_d*(1-b_d);
        float w001 = (1-r_d)*(1-g_d)*b_d;
        float w101 = r_d*(1-g_d)*b_d;
        float w011 = (1-r_d)*g_d*b_d;
        float w111 = r_d*g_d*b_d;

		int x_dst = position[1];
		int y_dst = position[0];

        img_enhance.at<cv::Vec3f>(y_dst, x_dst)[0] = w000 * r_lut[id000] + 
                                                     w100 * r_lut[id100] +
                                                     w010 * r_lut[id010] + 
                                                     w110 * r_lut[id110] +
                                                     w001 * r_lut[id001] + 
                                                     w101 * r_lut[id101] +
                                                     w011 * r_lut[id011] + 
                                                     w111 * r_lut[id111];

        img_enhance.at<cv::Vec3f>(y_dst, x_dst)[1] = w000 * g_lut[id000] + 
                                                     w100 * g_lut[id100] +
                                                     w010 * g_lut[id010] + 
                                                     w110 * g_lut[id110] +
                                                     w001 * g_lut[id001] + 
                                                     w101 * g_lut[id101] +
                                                     w011 * g_lut[id011] + 
                                                     w111 * g_lut[id111];

        img_enhance.at<cv::Vec3f>(y_dst, x_dst)[2] = w000 * b_lut[id000] + 
                                                     w100 * b_lut[id100] +
                                                     w010 * b_lut[id010] + 
                                                     w110 * b_lut[id110] +
                                                     w001 * b_lut[id001] + 
                                                     w101 * b_lut[id101] +
                                                     w011 * b_lut[id011] + 
                                                     w111 * b_lut[id111];

        // img_enhance.at<cv::Vec3f>(y_dst, x_dst)[0] = w000 * r_lut.at<float>(r_id, g_id, b_id) + 
        //                                              w100 * r_lut.at<float>(r_id + 1, g_id, b_id) +
        //                                              w010 * r_lut.at<float>(r_id, 1 + g_id, b_id) + 
        //                                              w110 * r_lut.at<float>(r_id + 1, 1 + g_id, b_id) +
        //                                              w001 * r_lut.at<float>(r_id, g_id, b_id + 1) + 
        //                                              w101 * r_lut.at<float>(r_id, g_id, b_id + 1) +
        //                                              w011 * r_lut.at<float>(r_id, g_id + 1, b_id + 1) + 
        //                                              w111 * r_lut.at<float>(r_id + 1, g_id + 1, b_id + 1);

        // img_enhance.at<cv::Vec3f>(y_dst, x_dst)[1] = w000 * g_lut.at<float>(r_id, g_id, b_id) + 
        //                                              w100 * g_lut.at<float>(r_id + 1, g_id, b_id) +
        //                                              w010 * g_lut.at<float>(r_id, 1 + g_id, b_id) + 
        //                                              w110 * g_lut.at<float>(r_id + 1, 1 + g_id, b_id) +
        //                                              w001 * g_lut.at<float>(r_id, g_id, b_id + 1) + 
        //                                              w101 * g_lut.at<float>(r_id, g_id, b_id + 1) +
        //                                              w011 * g_lut.at<float>(r_id, g_id + 1, b_id + 1) + 
        //                                              w111 * g_lut.at<float>(r_id + 1, g_id + 1, b_id + 1);

        // img_enhance.at<cv::Vec3f>(y_dst, x_dst)[2] = w000 * b_lut.at<float>(r_id, g_id, b_id) + 
        //                                              w100 * b_lut.at<float>(r_id + 1, g_id, b_id) +
        //                                              w010 * b_lut.at<float>(r_id, 1 + g_id, b_id) + 
        //                                              w110 * b_lut.at<float>(r_id + 1, 1 + g_id, b_id) +
        //                                              w001 * b_lut.at<float>(r_id, g_id, b_id + 1) + 
        //                                              w101 * b_lut.at<float>(r_id, g_id, b_id + 1) +
        //                                              w011 * b_lut.at<float>(r_id, g_id + 1, b_id + 1) + 
        //                                              w111 * b_lut.at<float>(r_id + 1, g_id + 1, b_id + 1);

		// img_enhance.at<cv::Vec3f>(y_dst, x_dst)[0] = r;
		// img_enhance.at<cv::Vec3f>(y_dst, x_dst)[1] = g;
		// img_enhance.at<cv::Vec3f>(y_dst, x_dst)[2] = b;        
    });

    return img_enhance * 255;
}