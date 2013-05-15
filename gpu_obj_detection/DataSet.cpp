/*
 * DataSet.cpp
 *
 *  Created on: Apr 28, 2013
 *      Author: olehp
 */

#include "DataSet.h"
#include "gpu_prepare_data_set.h"

DataSet::DataSet(const char * pos_list, const char * neg_list) {

	std::vector<LabeledImg> container;
	LoadImages(pos_list, neg_list, container);

	data.create(container.size(), GetFeatureSet().size());
	labels.create(container.size(), 1);

//	for (uint i = 0; i < container.size(); i++) {
//		LabeledImg& limg = container[i];
//		NormalizeMat(limg.first);
//		ToIntegralImage(limg.first, SINGLE_SUM);
//	}

	std::vector<Feature> & feature_set = GetFeatureSet();

	gpu_prepare_data_set(container, feature_set, labels, data);

//	for (uint i = 0; i < container.size(); i++) {
//		for (uint j = 0; j < feature_set.size(); j++) {
//			data(i, j) = feature_set[j].Eval(container[i].first);
//		}
//		labels(i, 0) = container[i].second;
//	}

	PrintMatrix((cv::Mat_<float>) data.colRange(0, 3));
}
