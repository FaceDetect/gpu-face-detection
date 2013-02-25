/*
 * TrainingData.h
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#ifndef TRAININGDATA_H_
#define TRAININGDATA_H_

#include <opencv2/core/core.hpp>
#include <vector>
#include "Feature.h"

class TrainingData {
public:
	TrainingData();
	void LoadImages(const char* pos_image_list_path,
			  	  	const char* neg_image_list_path);
	void PrepareDataSet();
	void CreateDataEntry(cv::Mat_<int> &ii, int *entry, int class_label);
	std::vector<cv::Mat_<int> > images_pos;
	std::vector<cv::Mat_<int> > images_neg;
	std::vector<cv::Mat_<int> > ii_pos;
	std::vector<cv::Mat_<int> > ii_neg;
	cv::Mat_<int> data_set;
	std::vector<Feature> features;
};

#endif /* TRAININGDATA_H_ */
