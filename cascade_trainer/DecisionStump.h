/*
 * DecisionTree.h
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#ifndef DECISIONSTUMP_H_
#define DECISIONSTUMP_H_

#include <opencv2/opencv.hpp>

class DecisionStump {
public:
	DecisionStump();
	DecisionStump(double threshold, int i, bool gt);

	static struct DecisionStumpInfo Build(cv::Mat_<int> &dataset, cv::Mat_<double> D);

	cv::Mat_<int> Classify(cv::Mat_<int> &dataset);

	void PrintInfo();

	double threshold;
	int i_feature;
	bool gt;

	static double WgError(const cv::Mat_<double>& err_arr,
			              const cv::Mat_<double> &D);

	static cv::Mat_<double> ErrorArr(const cv::Mat_<int>& predicted_vals,
									 const cv::Mat_<int>& class_labels);

};

struct DecisionStumpInfo {

	DecisionStumpInfo(const DecisionStump &stump,
					  const cv::Mat_<double> &err_arr,
					  const cv::Mat_<double> &best_pred,
					  double wg_err) {
		this->ds = stump;
		this->err_arr = err_arr;
		this->best_pred = best_pred;
		this->wg_err = wg_err;
	}

	DecisionStump ds;
	cv::Mat_<double> err_arr;
	cv::Mat_<double> best_pred;
	double wg_err;
};

#endif /* DECISIONSTUMP_H_ */
