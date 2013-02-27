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

#define GET_MAT_COL(mat, i_col) \
	(mat).colRange((i_col), (i_col) + 1)

#define ENDL std::cout << std::endl;

void GenerateFeatures(std::vector<Feature>& features);
cv::Mat_<int> ComputeIntegralImage(cv::Mat_<int> &mat);
void PrintMatrix(const cv::Mat_<int> &mat);


template <class InputIterator, class OutputIterator, class Transformator>
OutputIterator relaxed_transform(InputIterator first, InputIterator last,
                                 OutputIterator result, Transformator trans) {
	for (; first != last; ++first, ++result)
		*result = trans(*first);

	return result;
}

#endif /* UTILS_H_ */
