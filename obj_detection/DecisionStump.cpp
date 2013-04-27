/*
 * DecisionStump.cpp
 *
 *  Created on: Apr 26, 2013
 *      Author: olehp
 */

#include <climits>

#include "DecisionStump.h"

using namespace cv;


DecisionStump::DecisionStump() :
		threshold(-1),
		i_feature(-1),
		gt(false) {
}

DecisionStump::DecisionStump(int i, float threshold, bool gt) :
				threshold(threshold),
				i_feature(i),
				gt(gt) {
}

cv::Mat_<label_t> DecisionStump::Classify(Data data) {
	Mat_<label_t> labels(data.rows, 1);

	for (int i = 0; i < data.rows; i++) {
		labels(i, 0) = (label_t) ((gt) ? (data(i, i_feature) > threshold) : (data(i, i_feature) <= threshold));
	}

	return labels;
}

void DecisionStump::Train(DataSet& data_set) {
	float least_err = FLT_MAX;
//	cv::Mat_<double> best_err_arr;
//	cv::Mat_<int> best_pred;

	for (int col = 0; col < data_set.data.cols; col++) {

		Mat_<float> feature_vals = data_set.data.col(col);

		for (float &thr : feature_vals) {
			for (int ineq = 0; ineq <= 1; ineq++) {

				DecisionStump curr_stump(col, thr, ineq);

				Mat_<int> pred = curr_stump.Classify(data_set.data);
				Mat_<int> err_arr = data_set.labels.clone();

				compare(pred, data_set.labels, err_arr, CMP_NE);

				int curr_err = sum(err_arr).val[0];

				if (curr_err < least_err) {
					*this = curr_stump;
					least_err = curr_err;
				}
			}
		}
	}
}
