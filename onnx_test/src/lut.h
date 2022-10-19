//
// Created by ts on 2022/6/2.
//
#pragma once

#include <opencv2/core.hpp>

class Lut
{
public:
	static cv::Mat trilinear(const cv::Mat &image, const cv::Mat &lut);
    static void trilinear_forward(const float* lut, const cv::Mat &image, cv::Mat &output, const int dim=16);
private:
	/**
	 * http://www.paulbourke.net/miscellaneous/interpolation/
	 */
	// linear
	template<typename T, typename U>
	static T linearInterp(const T &p0, const T &p1, U &z)
	{
		return p0 * (1.0 - z) + p1 * z;
	}
	// bilinear
	template<typename T, typename U>
	static T linearInterp(const T &p00, const T &p01, const T &p10, const T &p11, U &y, U &z)
	{
		auto v1 = linearInterp(p00, p01, z);
		auto v2 = linearInterp(p10, p11, z);
		return linearInterp(v1, v2, y);
	}
	// trilinear
	template<typename T, typename U>
	static T linearInterp(const T &p000, const T &p001, const T &p010, const T &p011,
								 const T &p100, const T &p101, const T &p110, const T &p111, U &x, U &y, U &z)
	{
		auto v1 = linearInterp(p000, p001, p010, p011, y, z);
		auto v2 = linearInterp(p100, p101, p110, p111, y, z);
		return linearInterp(v1, v2, x);
	}

	// cosine
	template<typename T, typename U>
	static T cosineInterp(const T &p0, const T &p1, U &z)
	{
		U x2 = (1.0 - cos(z * CV_PI)) / 2.0;
		return (p0 * (1.0 - x2) + p1 * x2);
	}
	template<typename T, typename U>
	static T cosineInterp(const T &p00, const T &p01, const T &p10, const T &p11, U &y, U &z)
	{
		auto v1 = cosineInterp(p00, p01, z);
		auto v2 = cosineInterp(p10, p11, z);
		return cosineInterp(v1, v2, y);
	}
	template<typename T, typename U>
	static T cosineInterp(const T &p000, const T &p001, const T &p010, const T &p011,
								 const T &p100, const T &p101, const T &p110, const T &p111, U &x, U &y, U &z)
	{
		auto v1 = cosineInterp(p000, p001, p010, p011, y, z);
		auto v2 = cosineInterp(p100, p101, p110, p111, y, z);
		return cosineInterp(v1, v2, x);
	}

	/**
	 * https://www.paulinternet.nl/?page=bicubic
	 * http://www.paulbourke.net/miscellaneous/interpolation/
	 */
	// cubic
	template<typename T, typename U>
	static T cubicInterp(const T &p0_0, const T &p0, const T &p1, const T &p1_1, U &z)
	{
		T a0, a1, a2, a3;
		U z2 = z * z;
		a0 = -0.5 * p0_0 + 1.5 * p0 - 1.5 * p1 + 0.5 * p1_1;
		a1 = p0_0 - 2.5 * p0 + 2.0 * p1 - 0.5 * p1_1;
		a2 = -0.5 * p0_0 + 0.5 * p1_1;
		a3 = p0;
		//p0 + 0.5 * z * (p1 - p0_0 + z * (2.0 * p0_0 - 5.0 * p0 + 4.0 * p1 - p1_1 + z * (3.0 * (p0 - p1) + p1_1 - p0)))
		return (a0 * z * z2 + a1 * z2 + a2 * z + a3);
	}
	// bicubic 4 * 4
	template<typename T, typename U>
	static T cubicInterp(const T &p01_01, const T &p01_11, const T &p11_01, const T &p11_11,
								const T &p01_00, const T &p01_10, const T &p11_00, const T &p11_10,
								const T &p00_01, const T &p00_11, const T &p10_01, const T &p10_11,
								const T &p00_00, const T &p00_10, const T &p10_00, const T &p10_10, U &y, U &z)
	{
		auto v1 = cubicInterp(p00_00, p00_01, p01_00, p01_01, z);
		auto v2 = cubicInterp(p00_10, p00_11, p01_10, p01_11, z);
		auto v3 = cubicInterp(p10_00, p10_01, p11_00, p11_01, z);
		auto v4 = cubicInterp(p10_10, p10_11, p11_10, p11_11, z);
		return cubicInterp(v1, v2, v3, v4, y);
	}
	// tricubic
	template<typename T, typename U>
	static T cubicInterp(const T &p001_001, const T &p001_011, const T &p011_001, const T &p011_011,
								const T &p001_000, const T &p001_010, const T &p011_000, const T &p011_010,
								const T &p000_001, const T &p000_011, const T &p010_001, const T &p010_011,
								const T &p000_000, const T &p000_010, const T &p010_000, const T &p010_010,

								const T &p001_101, const T &p001_111, const T &p011_101, const T &p011_111,
								const T &p001_100, const T &p001_110, const T &p011_100, const T &p011_110,
								const T &p000_101, const T &p000_111, const T &p010_101, const T &p010_111,
								const T &p000_100, const T &p000_110, const T &p010_100, const T &p010_110,

								const T &p101_001, const T &p101_011, const T &p111_001, const T &p111_011,
								const T &p101_000, const T &p101_010, const T &p111_000, const T &p111_010,
								const T &p100_001, const T &p100_011, const T &p110_001, const T &p110_011,
								const T &p100_000, const T &p100_010, const T &p110_000, const T &p110_010,

								const T &p101_101, const T &p101_111, const T &p111_101, const T &p111_111,
								const T &p101_100, const T &p101_110, const T &p111_100, const T &p111_110,
								const T &p100_101, const T &p100_111, const T &p110_101, const T &p110_111,
								const T &p100_100, const T &p100_110, const T &p110_100, const T &p110_110,

								U &x, U &y, U &z)
	{
		auto v1 = cubicInterp(p001_001, p001_011, p011_001, p011_011,
							  p001_000, p001_010, p011_000, p011_010,
							  p000_001, p000_011, p010_001, p010_011,
							  p000_000, p000_010, p010_000, p010_010, y, z);

		auto v2 = cubicInterp(p001_101, p001_111, p011_101, p011_111,
							  p001_100, p001_110, p011_100, p011_110,
							  p000_101, p000_111, p010_101, p010_111,
							  p000_100, p000_110, p010_100, p010_110, y, z);

		auto v3 = cubicInterp(p101_001, p101_011, p111_001, p111_011,
							  p101_000, p101_010, p111_000, p111_010,
							  p100_001, p100_011, p110_001, p110_011,
							  p100_000, p100_010, p110_000, p110_010, y, z);

		auto v4 = cubicInterp(p101_101, p101_111, p111_101, p111_111,
							  p101_100, p101_110, p111_100, p111_110,
							  p100_101, p100_111, p110_101, p110_111,
							  p100_100, p100_110, p110_100, p110_110, y, z);
		return cubicInterp(v1, v2, v3, v4, x);
	}
	/**
	 * Tetrahedral
	 * from OpenColorIO
	 * https://github.com/AcademySoftwareFoundation/OpenColorIO/blob/master/src/OpenColorIO/ops/lut3d/Lut3DOpCPU.cpp
	 * http://ijetch.org/papers/318-T860.pdf
	 */
	template<typename T, typename U>
	static T tetrahedralInterp(const T &p000, const T &p001, const T &p010, const T &p011,
									  const T &p100, const T &p101, const T &p110, const T &p111, U &x, U &y, U &z)
	{
		if (x > y) {
			if (y > z) {
				return (1.0 - x) * p000 + (x - y) * p100 + (y - z) * p110 + z * p111;
			} else if (x > z) {
				return (1.0 - x) * p000 + (x - z) * p100 + (z - y) * p101 + y * p111;
			} else {
				return (1.0 - z) * p000 + (z - x) * p001 + (x - y) * p101 + y * p111;
			}
		} else {
			if (z > y) {
				return (1.0 - z) * p000 + (z - y) * p001 + (y - x) * p011 + x * p111;
			} else if (z > x) {
				return (1.0 - y) * p000 + (y - z) * p010 + (z - x) * p011 + x * p111;
			} else {
				return (1.0 - y) * p000 + (y - x) * p010 + (x - z) * p110 + z * p111;
			}
		}
	}
};

