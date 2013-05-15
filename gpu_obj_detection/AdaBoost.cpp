/*
 * AdaBoost.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#include <cmath>
#include <algorithm>

#include "AdaBoost.h"
#include "utils.h"
#include "constants.h"

using namespace cv;
using namespace std;

AdaBoost::AdaBoost(DataSet& data_set) :
	data_set(data_set),
	threshold(0) {
	InitWeights();
}

//
//void AdaBoost::Train(int num_stumps) {
//
//	Mat_<double> D = Mat_<double>::ones(data.num_total, 1) / data.num_total;
//
//	Mat_<int> class_labels = GET_MAT_COL(data.data_set, data.data_set.cols - 1);
//
//	for (int t = 0; t < num_stumps; ++t) {
//
//		NormalizeWeights(D);
//
//		cout << "T = " << t << endl;
//
//		DecisionStumpInfo info = DecisionStump::Build(data.data_set, D);
//
//		info.ds.PrintInfo();
//
//
//		double beta = info.wg_err / (1 - info.wg_err);
//
//		UpdateWeights(D, info.err_arr, beta);
//
//		BoostStump bs(info.ds);
//
//		thresold += 0.5 * (bs.alpha = CalcAlpha(beta));
//
//		stumps.push_back(bs);
//
//		double err_rate = sum(DecisionStump::ErrorArr(Classify(data), class_labels)).val[0];
//
//		cout << "TOTAL ERR RATE: " << err_rate / class_labels.rows << endl;
//
//		if (err_rate == 0)
//			break;
//	}
//}
//
//cv::Mat_<int> AdaBoost::Classify(TrainingData &td) {
//	Mat_<double> product = Mat_<double>::zeros(td.data_set.rows, 1);
//	Mat_<int> res(td.data_set.rows, 1);
//
//	for (uint i = 0; i < stumps.size(); ++i) {
//		product += stumps.at(i).Classify(td.data_set);
//	}
//
//	for (int i = 0; i < res.rows; ++i) {
//		res(i, 0) = (product(i, 0) >= thresold) ? POSITIVE_LABEL : NEGATIVE_LABEL;
//	}
//
//	return res;
//}
//
//void AdaBoost::UpdateWeights(cv::Mat_<double>& D, cv::Mat_<double>& err_arr,
//							 double beta) {
//
//	for (int i = 0; i < D.rows; ++i)
//		D(i, 0) *= pow(beta, 1 - err_arr(i, 0));
//}
//
//void AdaBoost::NormalizeWeights(cv::Mat_<double>& D) {
//	double total = sum(D).val[0];
//
//	if (total == 0) return;
//
//	for (int i = 0; i < D.rows; ++i) {
//		D(i, 0) /= total;
//	}
//}
//
//double AdaBoost::CalcAlpha(double beta) {
//	return log(1 / max(beta, 1e-10));
//}

void AdaBoost::TrainWeak() {
	float w_sum = sum(W).val[0];

	for (int i = 0; i < W.rows; i++)
		W(i, 0) /= w_sum;

	DecisionStump stump;
	stump.Train(data_set, W);

	float beta = stump.wg_err / (1 - stump.wg_err);

	for (int i = 0; i < W.rows; i++)
		W(i, 0) *= pow(beta, 1 - stump.err_arr(i, 0));

	float alpha = log(1 / beta);

	this->threshold += 0.5 * alpha;

	stumps.push_back(make_pair(stump, alpha));
}

std::vector<std::pair<DecisionStump, float> >& AdaBoost::GetStumps() {
	return stumps;
}

void AdaBoost::Classify(const Data& data, cv::Mat_<label_t>& labels) {
	Mat_<float> prod = Mat_<float>::zeros(data.rows, 1);
	labels.create(data.rows, 1);

	for (uint i = 0; i < stumps.size(); i++) {
		Mat_<label_t> labels_inner;
		stumps[i].first.Classify(data, labels_inner);
		prod += ((Mat_<float>)labels_inner * stumps[i].second);
	}

	for (int i = 0; i < labels.rows; i++) {
		labels(i, 0) = prod(i, 0) >= threshold;
	}
}

void AdaBoost::Clear() {
	stumps.clear();
	InitWeights();
}

void AdaBoost::InitWeights() {
	W.create(data_set.labels.rows, 1);

	int num_pos = count(data_set.labels.begin(), data_set.labels.end(), POSITIVE_LABEL);
	int num_neg = count(data_set.labels.begin(), data_set.labels.end(), NEGATIVE_LABEL);

	for (int i = 0; i < data_set.labels.rows; i++) {
		W(i, 0) = (data_set.labels(i, 0) == POSITIVE_LABEL) ? (1 / (2.0 * num_pos)) : (1 / (2.0 * num_neg));
	}
}
