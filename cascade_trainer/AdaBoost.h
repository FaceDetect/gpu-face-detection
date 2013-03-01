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

#include "BoostStump.h"
#include "TrainingData.h"

class AdaBoost {

public:
	AdaBoost(TrainingData &data);
	void Train(int num_stumps);
	cv::Mat_<int> Classify(TrainingData &td);
	double thresold;
private:
	void NormalizeWeights(cv::Mat_<double> &D);
	void UpdateWeights(cv::Mat_<double> &D, cv::Mat_<double> &err_arr, double beta);
	double CalcAlpha(double beta);
	TrainingData &data;
	std::vector<BoostStump> stumps;
};

#endif /* ADABOOST_H_ */
