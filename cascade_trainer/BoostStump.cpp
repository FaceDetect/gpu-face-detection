/*
 * BoostStump.cpp
 *
 *  Created on: Feb 27, 2013
 *      Author: olehp
 */

#include "BoostStump.h"

using namespace cv;


BoostStump::BoostStump(const DecisionStump& ds) {
	threshold = ds.threshold;
	i_feature = ds.i_feature;
	gt = ds.gt;
	alpha = 0;
}

cv::Mat_<double> BoostStump::Classify(cv::Mat_<int>& dataset) {

	return (alpha == 0) ? Mat_<double>::zeros(dataset.rows, 1) :(Mat_<double>)DecisionStump::Classify(dataset) * alpha;

}
