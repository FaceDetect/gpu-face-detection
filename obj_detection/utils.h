/*
 * utils.h
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <termios.h>
#include <unistd.h>
#include <stdio.h>

#include "Feature.h"


#define GET_MAT_COL(mat, i_col) \
	(mat).colRange((i_col), (i_col) + 1)

#define ENDL std::cout << std::endl;
#define SINGLE_SUM 0
#define SQUARED_SUM 1
#define SQR(a) (( a ) * ( a ))

int mygetch();
void GenerateFeatures(std::vector<Feature>& features);
cv::Mat_<int> ComputeIntegralImage(const cv::Mat_<int> &mat, int mode);

template <class T>
bool MatrEq(const cv::Mat_<T> &mat1, const cv::Mat_<T> &mat2) {
	if ((mat1.rows != mat2.rows) || (mat1.cols != mat2.cols)) return false;

	cv::MatConstIterator_<T> it1 = mat1.begin();
	cv::MatConstIterator_<T> it2 = mat2.begin();

	for (; it1 != mat1.end(); it1++, it2++)
		if (*it1 != *it2) return false;

	return true;
}


template <class T>
void PrintMatrix(const cv::Mat_<T> &mat) {
	for (int row = 0; row < mat.rows; row++) {
		for (int col = 0; col < mat.cols; col++) {
			std::cout << mat(row, col) << "\t";
		}
		std::cout << std::endl;
	}
}


template <class InputIterator, class OutputIterator, class Transformator>
OutputIterator relaxed_transform(InputIterator first, InputIterator last,
                                 OutputIterator result, Transformator trans) {
	for (; first != last; ++first, ++result)
		*result = trans(*first);

	return result;
}

#endif /* UTILS_H_ */
