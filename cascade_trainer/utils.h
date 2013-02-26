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


#endif /* UTILS_H_ */
