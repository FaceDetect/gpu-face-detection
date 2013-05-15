/*
 * DataSet.h
 *
 *  Created on: Apr 21, 2013
 *      Author: olehp
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <opencv2/opencv.hpp>

#include "utils.h"
#include "constants.h"

typedef cv::Mat_<float> Data;

struct DataSet {
	DataSet() { }
	DataSet(const char * pos_list, const char * neg_list);

	DataSet(Data & data, cv::Mat_<label_t> & labels) {
		this->data = data;
		this->labels = labels;
	}


	Data data;
	cv::Mat_<label_t> labels;
};

#endif /* DATASET_H_ */
