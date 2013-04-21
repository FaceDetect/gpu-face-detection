/*
 * DataSet.h
 *
 *  Created on: Apr 21, 2013
 *      Author: olehp
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <opencv2/opencv.hpp>

typedef cv::Mat_<int> Data;

struct DataSet {
	Data data;
	cv::Mat_<int> labels;
};

#endif /* DATASET_H_ */
