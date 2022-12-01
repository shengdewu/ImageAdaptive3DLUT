#include <iostream>
#include <fstream>
#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <assert.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <opencv2/core/simd_intrinsics.hpp>
#include "enhance.h"
#include "onnx_enhance.h"


void bat_test(std::string file_list_txt, std::string root_path, std::string out_root){
    std::ifstream ifs;
    ifs.open(file_list_txt, std::ios::in);
    if(!ifs.is_open()){
        std::cout << file_list_txt << std::endl;
        return;
    }

    std::string mnn_path = "/mnt/sda1/workspace/enhance/ImageAdaptive3DLUT/onnx_test/lut.mnn";
    ImgEnhance img_enhance(mnn_path, 8);

    std::string split("#");
    std::string ctrl_n("\n");
    std::string line;
    while(std::getline(ifs, line)){
        std::cout << line << std::endl;
        size_t pos = line.find(split);
        std::string sub_path(line.substr(pos+1, line.size()-ctrl_n.size()));
        std::cout << sub_path << std::endl;
        std::string out_path = out_root + "/" + sub_path;
        std::string cmd("mkdir -p " + std::string("\"") + out_path + std::string("\""));
        std::cout << cmd << std::endl;
        int ret = system(cmd.c_str());
        if(ret){
            std::cout << strerror(errno) << std::endl;
        }
        std::string img_path = root_path + "/" + sub_path;
        struct dirent* ptr;
        DIR *dir = opendir(img_path.c_str());
        while((ptr=readdir(dir)) != NULL){
            if(strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0){
                continue;
            }

            std::string name = ptr->d_name;

            size_t pos = name.rfind('.');
            std::string out_name = out_path + "/" + name.substr(0, pos)+ ".jpg";
            // std::cout << out_name << std::endl;
            std::ifstream f(out_name.c_str());

            if (f.good()){
                std::cout << out_name << " has found "<< std::endl;
                continue;
            }

            std::string img_name = img_path + "/" + name ;

            cv::Mat img_bgr = cv::imread(img_name, cv::ImreadModes::IMREAD_COLOR);
            cv::Mat img_rgb;
            cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

            cv::Mat enhance_img = img_enhance.run(img_rgb, 512);

            cv::Mat enhance_img_bgr;
            cv::cvtColor(enhance_img, enhance_img_bgr, cv::COLOR_RGB2BGR);

            cv::Mat concat;
            cv::hconcat(img_bgr, enhance_img_bgr, concat);

            cv::imwrite(out_name, concat);
        }
    }
}



int main(int argc, char** argv){
    // bat_test("/mnt/sda1/workspace/enhance/ImageAdaptive3DLUT/dir/error.txt", "/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif_3000x2000", "/mnt/sda1/valid.output/enhance.test/lut_mnn_quan");

    std::string mnn_path = "/mnt/sda1/workspace/enhance/ImageAdaptive3DLUT/lut.mnn";
//    std::string onnx_path = "/mnt/sda1/workspace/ImageAdaptive3DLUT/onnx_test/lut.onnx";
    std::string out_path = "/mnt/sda1/workspace/enhance/ImageAdaptive3DLUT/test.out/";

    auto init_time = std::chrono::system_clock::now();

//    OnnxImgEnhance img_enhance(onnx_path, 8);

    ImgEnhance img_enhance(mnn_path, 8);

    std::vector<std::string> img_path;
    // img_path.push_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif/儿童/01浅色实景/儿童SJ (31)/161082062385739907252.tif");
    // img_path.push_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif/儿童/01浅色实景/儿童SJ (31)/16108206392879900290.tif");
//    img_path.push_back("/mnt/sda1/workspace/ximg/test/DSC_4678.jpg");
    img_path.push_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif_3000x2000/test/1.debug.raw.jpg");
    img_path.push_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif_3000x2000/test/2.debug.raw.jpg");
    img_path.push_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif_3000x2000/test/debug.raw.jpg");
    img_path.push_back("/mnt/sda1/workspace/enhance/ImageAdaptive3DLUT/test.img/debug.raw.jpg");
    
    for(size_t i=0; i < img_path.size(); i++){
        size_t spos = img_path[i].rfind('/')+1;
        size_t epos = img_path[i].find(".jpg");
        std::string name = img_path[i].substr(spos, epos-spos);

        auto read_time = std::chrono::system_clock::now();
        cv::Mat img_bgr = cv::imread(img_path[i], cv::ImreadModes::IMREAD_COLOR);
        cv::Mat img_rgb;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
        // cv::Mat img_bgr_normal, tmp;
        // img_bgr.convertTo(img_bgr_normal, CV_32FC3, 1.0/65535.0);
        // double min_val, max_val;
        // cv::minMaxLoc(img_bgr_normal, &min_val, &max_val);
        // std::cout << min_val << "," << max_val << std::endl;
        // auto en_time = std::chrono::system_clock::now();

        cv::Mat enhance_img = img_enhance.run(img_rgb, 540, out_path + name + ".lut.jpg", false);

        cv::Mat enhance_img_bgr;
        cv::cvtColor(enhance_img, enhance_img_bgr, cv::COLOR_RGB2BGR);

        // auto convert_time = std::chrono::system_clock::now();
        // cv::Mat enhance_255 =(enhance_img*255);
        // cv::Mat enhance_8u3c;
        // enhance_255.convertTo(enhance_8u3c, CV_8UC3);
        // cv::Mat img_bgr_255 = (img_bgr_normal*255);
        // cv::Mat img_bgr_8uc3;
        // img_bgr_255.convertTo(img_bgr_8uc3, CV_8UC3);
        // spend_time("convert time ", convert_time);
        // cv::Mat concat;
        // cv::hconcat(img_bgr, enhance_img_bgr, concat);
        // cv::hconcat(concat, onnx_enhance_img_bgr, concat);
        // size_t spos = img_path[i].rfind('/')+1;
        // size_t epos = img_path[i].find(".tif");
        // cv::imwrite(img_path[i].substr(spos, epos-spos)+".jpg", concat);
        cv::imwrite(out_path + name + ".jpg", enhance_img_bgr);
    }

}