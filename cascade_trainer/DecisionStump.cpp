/*
 * DecisionTree.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#include "DecisionStump.h"

using namespace cv;
using namespace std;


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

DecisionStumpInfo DecisionStump::Build(Mat_<int> &dataset, Mat_<double> D) {

	DecisionStump best_stump;
	double least_wg_err = DBL_MAX;
	cv::Mat_<double> best_err_arr;
	cv::Mat_<int> best_pred;

	Mat_<int> class_labels = GET_MAT_COL(dataset, dataset.cols - 1);

	for (int col = 0; col < dataset.cols - 1; col++) {

		if (col % 20000 == 0) cout << "Passed features: " << col << "/" << (dataset.cols - 1) << endl;

		Mat_<int> feature_vals = GET_MAT_COL(dataset, col);

		for (int &thr : feature_vals) {
			for (int ineq = 0; ineq <= 1; ineq++) {

				DecisionStump curr_stump(thr, col, ineq);
				Mat_<int> pred = curr_stump.Classify(dataset);
				Mat_<double> err_arr = ErrorArr(pred, class_labels);

				double wg_err = WgError(err_arr, D);

				if (wg_err < least_wg_err) {
					best_stump = curr_stump;
					least_wg_err = wg_err;
					best_err_arr = err_arr;
					best_pred = pred;
				}
			}
		}
	}
	return DecisionStumpInfo(best_stump, best_err_arr, best_pred, least_wg_err);
}

void DecisionStump::PrintInfo() {
	cout << "******THRESOLD INFO******" << endl;
	cout << "Threshold: " << threshold << endl;
	cout << "Feature id: " << i_feature<< endl;
	cout << "gt: " << gt << endl;
	cout << "*************************" << endl;
}

Mat_<double> DecisionStump::ErrorArr(const cv::Mat_<int>& predicted_vals,
									 const cv::Mat_<int>& class_labels) {
	Mat_<double> err_arr(class_labels.rows, 1);

	for (int row = 0; row < class_labels.rows; row++)
		err_arr(row, 0) = (class_labels(row, 0) == (int)predicted_vals(row, 0)) ? 0 : 1;

	return err_arr;
}

double DecisionStump::WgError(const cv::Mat_<double>& err_arr,
		                      const cv::Mat_<double> &D) {

	double wg_err = sum(err_arr.mul(D)).val[0];
	return wg_err;
}

cv::Mat_<int> DecisionStump::Classify(cv::Mat_<int>& dataset) {

	Mat_<double> result(dataset.rows, 1);
	Mat_<int> vals = GET_MAT_COL(dataset, i_feature);

	relaxed_transform(vals.begin(), vals.end(), result.begin(),
			[this](const int &val) -> double {
					return (gt) ? ((val > this->threshold) ? NEGATIVE_LABEL : POSITIVE_LABEL)
					 	 	 	: ((val <= this->threshold) ? NEGATIVE_LABEL : POSITIVE_LABEL); });
	return result;
}
