/*
 * test_utils.h
 *
 *  Created on: Apr 20, 2013
 *      Author: olehp
 */

#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <opencv2/opencv.hpp>

template<typename T>
cv::Mat_<T> CreateMatrix(int r, int c, T (*init_func)() = []() { static int i = 0; return ++i; }) {
	cv::Mat_<T> mat(r, c);

	for (T &iter : mat) iter = init_func();

	return mat;
}


#endif /* TEST_UTILS_H_ */
