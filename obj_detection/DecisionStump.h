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


	void Classify(const Data& data, cv::Mat_<label_t>& labels);

	void Train(const DataSet & data_set, const cv::Mat_<float> W);
	float threshold;
	int i_feature;
	bool gt;

private:
	float wg_err;
	cv::Mat_<float> err_arr;
};

#endif /* DECISIONSTUMP_H_ */
