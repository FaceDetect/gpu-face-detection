/*
 * CascadeTrainer.cpp
 *
 *  Created on: Apr 29, 2013
 *      Author: olehp
 */

#include "CascadeTrainer.h"

using namespace std;
using namespace cv;


CascadeTrainer::CascadeTrainer(DataSet& training_set, DataSet& test_set) :
	training_set(training_set),
	test_set(test_set){
}

void CascadeTrainer::Train(float f, float d, float f_target) {

	float F[1000];
	float D[1000];

	F[0] = 1.0;
	D[0] = 1.0;

	int i = 0;

	while (F[i] > f_target) {
		i++;
		F[i] = F[i - 1];

		int n = 0;

		AdaBoost *stage = new AdaBoost(training_set);
		stages.push_back(stage);

		while (F[i] > f * F[i - 1]) {
			n++;

			stage->Clear();

			for (int i = 0; i < n; i++)
				stage->TrainWeak();

			Mat_<label_t> labels;
			Classify(test_set.data, labels);

			// Discrease threshold

		}

		if (F[i] > f_target) {
			// Adjust negative set
		}
	}
}

void CascadeTrainer::Classify(const Data& data, cv::Mat_<label_t>& labels) {

	labels.create(data.rows, 1);
	for (int i = 0; i < labels.rows; i++) labels(i, 0) = POSITIVE_LABEL;

	for (int i = 0; i < data.rows; i++) {
		for (uint j = 0; j < stages.size(); j++) {
			Mat_<label_t> labels_inner;
			stages[j]->Classify(data.row(i), labels_inner);

			if (labels_inner(0, 0) == NEGATIVE_LABEL) {
				labels(i, 0) = NEGATIVE_LABEL;
				break;
			}
		}
	}
}
