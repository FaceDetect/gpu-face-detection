/*
 * DataSet.h
 *
 *  Created on: Apr 21, 2013
 *      Author: olehp
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <opencv2/opencv.hpp>

#include "utils.h"
#include "constants.h"

typedef cv::Mat_<float> Data;

struct DataSet {
	DataSet(const char * pos_list, const char * neg_list) {
		std::vector<LabeledImg> container;
		LoadImages(pos_list, neg_list, container);

		data.create(container.size(), GetFeatureSet().size());
		labels.create(container.size(), 1);

		for (LabeledImg &limg : container) {
			NormalizeMat(limg.first);
			ToIntegralImage(limg.first, SINGLE_SUM);
		}

		std::vector<Feature> & feature_set = GetFeatureSet();

		for (uint i = 0; i < container.size(); i++) {
			for (uint j = 0; j < feature_set.size(); j++) {
				data(i, j) = feature_set[j].Eval(container[i].first);
			}
			labels(i, 0) = container[i].second;
		}
	}

	DataSet(Data & data, cv::Mat_<label_t> & labels) {
		this->data = data;
		this->labels = labels;
	}


	Data data;
	cv::Mat_<label_t> labels;
};

#endif /* DATASET_H_ */
