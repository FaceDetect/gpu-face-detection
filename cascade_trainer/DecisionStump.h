/*
 * DecisionTree.h
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#ifndef DECISIONTREE_H_
#define DECISIONTREE_H_

#include <opencv2/opencv.hpp>

class DecisionStump {
public:
	DecisionStump();
	DecisionStump(double threshold, int i, bool gt);
	cv::Mat_<int> Classify(cv::Mat_<int> &dataset);
	double threshold;
	int i_feature;
	bool gt;

private:
	static DecisionStump Build(cv::Mat_<int> &dataset, cv::Mat_<double> D);
	static double WgError(const cv::Mat_<int> &predicted_vals,
						  const cv::Mat_<int> &class_labels,
						  const cv::Mat_<double> &D);

};

#endif /* DECISIONTREE_H_ */
