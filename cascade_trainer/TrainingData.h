/*
 * TrainingData.h
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#ifndef TRAININGDATA_H_
#define TRAININGDATA_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "Feature.h"

class TrainingData {
public:
	TrainingData();
	void LoadImages(const char* pos_image_list_path,
			  	  	const char* neg_image_list_path);
	void PrepareDataSet();
	void CreateDataEntry(cv::Mat_<int> &ii, int *entry, int class_label);
	void ShowImages();
	std::vector<cv::Mat_<int> > images_pos;
	std::vector<cv::Mat_<int> > images_neg;
	std::vector<cv::Mat_<int> > ii_pos;
	std::vector<cv::Mat_<int> > ii_neg;
	cv::Mat_<int> data_set;
	std::vector<Feature> features;

	int num_pos;
	int num_neg;
	int num_total;
	int num_features;
};

#endif /* TRAININGDATA_H_ */
