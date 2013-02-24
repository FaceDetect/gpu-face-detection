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

class TrainingData {
public:
	void LoadImages(const char *image_list_path);

	std::vector<cv::Mat> images;
};

#endif /* TRAININGDATA_H_ */
