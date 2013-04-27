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

class DecisionStump {
public:
	DecisionStump();
	DecisionStump(int i, float threshold, bool gt);
	cv::Mat_<label_t> Classify(Data data);
	void Train(DataSet & data_set);
	float threshold;
	int i_feature;
	bool gt;
};

#endif /* DECISIONSTUMP_H_ */
