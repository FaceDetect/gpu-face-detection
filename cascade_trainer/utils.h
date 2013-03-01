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
#include "Feature.h"
#include <termios.h>
#include <unistd.h>

#define GET_MAT_COL(mat, i_col) \
	(mat).colRange((i_col), (i_col) + 1)

#define ENDL std::cout << std::endl;

int mygetch();

void GenerateFeatures(std::vector<Feature>& features);
cv::Mat_<int> ComputeIntegralImage(cv::Mat_<int> &mat);

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
