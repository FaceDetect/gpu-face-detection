/*
 * BoostStump.h
 *
 *  Created on: Feb 27, 2013
 *      Author: olehp
 */

#ifndef BOOSTSTUMP_H_
#define BOOSTSTUMP_H_

#include "DecisionStump.h"

class BoostStump : public DecisionStump {
public:
	BoostStump(const DecisionStump &ds);
	cv::Mat_<double> Classify(cv::Mat_<int> &dataset);
	double alpha;
};

#endif /* BOOSTSTUMP_H_ */
