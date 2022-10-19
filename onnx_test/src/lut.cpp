//
// Created by ts on 2022/6/2.
//
#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "lut.h"


/// 空间换时间，减少循环中的判断

cv::Mat Lut::trilinear(const cv::Mat &src, const cv::Mat &lut)
{
	int index, cell;
	if (lut.cols == 64) {
		cell = 4; index = 15;
	} else if (lut.cols == 512) {
		cell = 8; index = 63;
	} else {
		std::cout << "lut size not support :" << lut.cols << " x" << lut.rows << std::endl;
		return src;
	}

    cv::Mat output = src.clone();
	output.forEach<cv::Vec3f>([&](cv::Vec3f &rgb, const int position[]) -> void {
		const float r = rgb[0] * (float) index;
		const float g = rgb[1] * (float) index;
		const float b = rgb[2] * (float) index;
		const int r0 = (int) r;
		const int g0 = (int) g;
		const int b0 = (int) b;
		const int r1 = std::min(r0 + 1, index);
		const int g1 = std::min(g0 + 1, index);
		const int b1 = std::min(b0 + 1, index);
		const float rd = r - (float) r0;
		const float gd = g - (float) g0;
		const float bd = b - (float) b0;

		// b = bx + by * 8  , x = r + bx * 64, y = g + by * 64
		// x = r + (b % 8) * 64, y = g + (b / 8) * 64
		const int len = index + 1;
		const int x000 = r0 + (b0 % cell) * len;
		const int y000 = g0 + (b0 / cell) * len;
		const int x100 = r0 + (b1 % cell) * len;
		const int y100 = g0 + (b1 / cell) * len;
		const int x010 = r0 + (b0 % cell) * len;
		const int y010 = g1 + (b0 / cell) * len;
		const int x110 = r0 + (b1 % cell) * len;
		const int y110 = g1 + (b1 / cell) * len;
		const int x001 = r1 + (b0 % cell) * len;
		const int y001 = g0 + (b0 / cell) * len;
		const int x101 = r1 + (b1 % cell) * len;
		const int y101 = g0 + (b1 / cell) * len;
		const int x011 = r1 + (b0 % cell) * len;
		const int y011 = g1 + (b0 / cell) * len;
		const int x111 = r1 + (b1 % cell) * len;
		const int y111 = g1 + (b1 / cell) * len;

		const auto ret = linearInterp(lut.at<cv::Vec3f>(y000, x000),
									  lut.at<cv::Vec3f>(y001, x001),
									  lut.at<cv::Vec3f>(y010, x010),
									  lut.at<cv::Vec3f>(y011, x011),
									  lut.at<cv::Vec3f>(y100, x100),
									  lut.at<cv::Vec3f>(y101, x101),
									  lut.at<cv::Vec3f>(y110, x110),
									  lut.at<cv::Vec3f>(y111, x111),
									  bd, gd, rd);

		const int y = position[0];
		const int x = position[1];
		rgb = ret;
	});

	return output;
}

void Lut::trilinear_forward(const float* lut, const cv::Mat &image, cv::Mat &output, const int dim)
{
    const int output_size = output.cols * output.rows;
    int shift = dim * dim * dim;
    const float binsize = 1.000001 / (dim-1);

    float *image_ptr = reinterpret_cast<float*>(image.data);
    float *out_ptr = reinterpret_cast<float*>(output.data);
    int index = 0;
    for (index = 0; index < output_size; ++index)
    {
        float r = image_ptr[index * 3];
        float g = image_ptr[index * 3 + 1];
        float b = image_ptr[index * 3 + 2];

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


        out_ptr[index * 3]= w000 * lut[id000] + w100 * lut[id100] +
                            w010 * lut[id010] + w110 * lut[id110] +
                            w001 * lut[id001] + w101 * lut[id101] +
                            w011 * lut[id011] + w111 * lut[id111];

        out_ptr[index * 3 + 1] = w000 * lut[id000 + shift] + w100 * lut[id100 + shift] +
                                 w010 * lut[id010 + shift] + w110 * lut[id110 + shift] +
                                 w001 * lut[id001 + shift] + w101 * lut[id101 + shift] +
                                 w011 * lut[id011 + shift] + w111 * lut[id111 + shift];

        out_ptr[index * 3 + 2] = w000 * lut[id000 + shift * 2] + w100 * lut[id100 + shift * 2] +
                                 w010 * lut[id010 + shift * 2] + w110 * lut[id110 + shift * 2] +
                                 w001 * lut[id001 + shift * 2] + w101 * lut[id101 + shift * 2] +
                                 w011 * lut[id011 + shift * 2] + w111 * lut[id111 + shift * 2];
    }
}