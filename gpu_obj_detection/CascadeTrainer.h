/*
 * CascadeTrainer.h
 *
 *  Created on: Apr 29, 2013
 *      Author: olehp
 */

#ifndef CASCADETRAINER_H_
#define CASCADETRAINER_H_

#include "AdaBoost.h"

#include <vector>

class CascadeTrainer {
public:
	CascadeTrainer(DataSet& training_set, DataSet& test_set);
	void Train(std::vector<int> num_weaks);

	void Classify(const Data& data, cv::Mat_<label_t>& labels);
	~CascadeTrainer();
private:
	DataSet &training_set;
	DataSet &test_set;

	std::vector<AdaBoost *> stages;

};

#endif /* CASCADETRAINER_H_ */
