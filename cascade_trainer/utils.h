/*
 * utils.h
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include "Feature.h"

#define W_HEIGHT 24
#define W_WIDTH 24


void GenerateFeatures(std::vector<Feature>& features);
cv::Mat_<int> ComputeIntegralImage(cv::Mat_<int> &mat);

#endif /* UTILS_H_ */
