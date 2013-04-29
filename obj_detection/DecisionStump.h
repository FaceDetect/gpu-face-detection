/*
 * DecisionStump.h
 *
 *  Created on: Apr 26, 2013
 *      Author: olehp
 */

#ifndef DECISIONSTUMP_H_
#define DECISIONSTUMP_H_

#include "utils.h"
#include "DataSet.h"

class AdaBoost;

class DecisionStump {
	friend AdaBoost;
public:
	DecisionStump();
	DecisionStump(int i, float threshold, bool gt);

	cv::Mat_<label_t> Classify(Data& data);

	void Train(DataSet & data_set, cv::Mat_<float> W);
	float threshold;
	int i_feature;
	bool gt;

private:
	float wg_err;
	cv::Mat_<float> err_arr;
};

#endif /* DECISIONSTUMP_H_ */
