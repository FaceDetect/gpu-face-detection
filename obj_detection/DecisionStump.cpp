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
		gt(false),
		wg_err(FLT_MAX) {
}

DecisionStump::DecisionStump(int i, float threshold, bool gt) :
				threshold(threshold),
				i_feature(i),
				gt(gt),
				wg_err(FLT_MAX) {
}

void DecisionStump::Classify(const Data& data, cv::Mat_<label_t>& labels) {
	labels.create(data.rows, 1);

	for (int i = 0; i < data.rows; i++) {
		labels(i, 0) = (label_t) ((gt) ? (data(i, i_feature) > threshold) : (data(i, i_feature) <= threshold));
	}
}

void DecisionStump::Train(const DataSet& data_set, const cv::Mat_<float> W) {

	for (int col = 0; col < data_set.data.cols; col++) {

		Mat_<float> feature_vals = data_set.data.col(col);

		for (float &thr : feature_vals) {
			for (int ineq = 0; ineq <= 1; ineq++) {

				DecisionStump curr_stump(col, thr, ineq);

				Mat_<int> pred;
				curr_stump.Classify(data_set.data, pred);
				Mat_<int> err_arr = data_set.labels.clone();

				for (int i = 0; i < err_arr.rows; i++) {
					err_arr(i, 0) = (pred(i, 0) != data_set.labels(i, 0));
				}

				float curr_err = sum(((Mat_<float>)err_arr).mul(W)).val[0];

				if (curr_err < this->wg_err) {
					*this = curr_stump;
					this->wg_err = curr_err;
					this->err_arr = err_arr.clone();
				}
			}
		}
	}
}
