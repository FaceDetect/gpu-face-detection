/*
 * AdaBoost.h
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

#include "DecisionStump.h"
#include "DataSet.h"

class AdaBoost {
public:

	// TODO: Define correct copy constructor.
	AdaBoost(DataSet& data_set);
	void TrainWeak();
	std::vector<std::pair<DecisionStump, float> >& GetStumps();

	void Classify(const Data& data, cv::Mat_<label_t>& labels);

	//TODO: Test case
	void Clear();

private:
//	void NormalizeWeights(cv::Mat_<double> &D);
//	void UpdateWeights(cv::Mat_<double> &D, cv::Mat_<double> &err_arr, double beta);
//	double CalcAlpha(double beta);
	void InitWeights();


	DataSet &data_set;
	float threshold;
	std::vector<std::pair<DecisionStump, float> > stumps;
	cv::Mat_<float> W;
};

#endif /* ADABOOST_H_ */
