/*
 * CascadeTrainer.h
 *
 *  Created on: Apr 29, 2013
 *      Author: olehp
 */

#ifndef CASCADETRAINER_H_
#define CASCADETRAINER_H_

#include "AdaBoost.h"

class CascadeTrainer {
public:
	CascadeTrainer(DataSet& training_set, DataSet& test_set);
	void Train(float f, float d, float f_target);

	void Classify(const Data& data, cv::Mat_<label_t>& labels);
private:
	DataSet &training_set;
	DataSet &test_set;

	std::vector<AdaBoost *> stages;

};

#endif /* CASCADETRAINER_H_ */
