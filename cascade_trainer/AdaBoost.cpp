/*
 * AdaBoost.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#include <cmath>

#include "AdaBoost.h"
#include "utils.h"
#include "constants.h"

using namespace cv;
using namespace std;


AdaBoost::AdaBoost(TrainingData &data) :
		thresold(0),
		data(data) {
}

void AdaBoost::Train(int num_stumps) {

	Mat_<double> D = Mat_<double>::ones(data.num_total, 1) / data.num_total;

	Mat_<int> class_labels = GET_MAT_COL(data.data_set, data.data_set.cols - 1);

	for (int t = 0; t < num_stumps; ++t) {

		NormalizeWeights(D);

		cout << "T = " << t << endl;

		DecisionStumpInfo info = DecisionStump::Build(data.data_set, D);

		info.ds.PrintInfo();


		double beta = info.wg_err / (1 - info.wg_err);

		UpdateWeights(D, info.err_arr, beta);

		BoostStump bs(info.ds);

		thresold += 0.5 * (bs.alpha = CalcAlpha(beta));

		stumps.push_back(bs);

		double err_rate = sum(DecisionStump::ErrorArr(Classify(data), class_labels)).val[0];

		cout << "TOTAL ERR RATE: " << err_rate / class_labels.rows << endl;

		if (err_rate == 0)
			break;
	}
}

cv::Mat_<int> AdaBoost::Classify(TrainingData &td) {
	Mat_<double> product = Mat_<double>::zeros(td.data_set.rows, 1);
	Mat_<int> res(td.data_set.rows, 1);

	for (uint i = 0; i < stumps.size(); ++i) {
		product += stumps.at(i).Classify(td.data_set);
	}

	for (int i = 0; i < res.rows; ++i) {
		res(i, 0) = (product(i, 0) >= thresold) ? POSITIVE_LABEL : NEGATIVE_LABEL;
	}

	return res;
}

void AdaBoost::UpdateWeights(cv::Mat_<double>& D, cv::Mat_<double>& err_arr,
							 double beta) {

	for (int i = 0; i < D.rows; ++i)
		D(i, 0) *= pow(beta, 1 - err_arr(i, 0));
}

void AdaBoost::NormalizeWeights(cv::Mat_<double>& D) {
	double total = sum(D).val[0];

	if (total == 0) return;

	for (int i = 0; i < D.rows; ++i) {
		D(i, 0) /= total;
	}
}

double AdaBoost::CalcAlpha(double beta) {
	return log(1 / max(beta, 1e-10));
}
