/*
 * DecisionTree.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#include "DecisionStump.h"

using namespace cv;

#include "constants.h"

#include <algorithm>
#include <climits>
#include "utils.h"

DecisionStump::DecisionStump() :
		threshold(-1),
		i_feature(-1),
		gt(false) {
}

DecisionStump::DecisionStump(double threshold, int i, bool gt) :
				threshold(threshold),
				i_feature(i),
				gt(gt) {
}

DecisionStump DecisionStump::Build(Mat_<int> &dataset, Mat_<double> D) {

	DecisionStump best_stump;
	double least_wg_err = DBL_MAX;
	Mat_<int> class_labels = dataset.colRange(dataset.cols, dataset.cols + 1);

	for (int col = 0; col < dataset.cols - 1; col++) {
		Mat_<int> feature_vals = dataset.colRange(col, col + 1);

		double max_val;
		double min_val;

		minMaxIdx(feature_vals, &min_val, &max_val);

		double step_size = (max_val - min_val) / NUM_STEPS;

		for (int step = 0; step < NUM_STEPS; step++) {

			DecisionStump curr_stump_gt(min_val + step * step_size, col, 1);
			DecisionStump curr_stump_lt(min_val + step * step_size, col, 0);

			double wg_err_gt = WgError(curr_stump_gt.Classify(dataset),
									   class_labels,
									   D);

			double wg_err_lt = WgError(curr_stump_lt.Classify(dataset),
					   	   	   	   	   class_labels,
					   	   	   	   	   D);

			if (wg_err_lt < least_wg_err) {
				best_stump = curr_stump_lt;
				least_wg_err = wg_err_lt;
			}

			if (wg_err_gt < least_wg_err) {
				best_stump = curr_stump_lt;
				least_wg_err = wg_err_gt;
			}
		}

	}
	return best_stump;
}

double DecisionStump::WgError(const cv::Mat_<int>& predicted_vals,
		                      const cv::Mat_<int>& class_labels,
		                      const cv::Mat_<double> &D) {


	Mat_<double> err_arr(class_labels.rows, 1);

	for (int row = 0; row < class_labels.rows; row++)
		err_arr(row, 0) = (class_labels(row, 0) == predicted_vals(row, 0)) ? 0 : 1;

	double wg_err = sum(err_arr.mul(D)).val[0];

	return wg_err;
}

cv::Mat_<int> DecisionStump::Classify(cv::Mat_<int>& dataset) {

	Mat_<int> result(dataset.rows, 1);
	Mat_<int> vals = GET_MAT_COL(dataset, i_feature);

	std::transform(vals.begin(), vals.end(), result.begin(),
			[this](const int &val) -> int {
					if (gt) return (val > this->threshold) ? 0 : 1;
					else return (val <= this->threshold) ? 0 : 1; });

	return result;
}
