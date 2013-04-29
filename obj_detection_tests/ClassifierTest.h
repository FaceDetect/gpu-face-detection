/*
 * ClassifierTest.h
 *
 *  Created on: Apr 28, 2013
 *      Author: olehp
 */

#ifndef CLASSIFIERTEST_H_
#define CLASSIFIERTEST_H_

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "MatTest.h"
#include "DataSet.h"
#include "utils.h"

class ClassifierTest: public MatTest {
public:
	virtual void SetUp() {
		Data data(3, 3);
		cv::Mat_<label_t> labels1(3, 1);
		cv::Mat_<label_t> labels2(3, 1);

		labels1(0, 0) = 0;
		labels1(1, 0) = 0;
		labels1(2, 0) = 1;

		labels2(0, 0) = 1;
		labels2(1, 0) = 1;
		labels2(2, 0) = 0;

		data(0, 0) = 1;
		data(0, 1) = 1;
		data(0, 2) = 1;
		data(1, 0) = 1;
		data(1, 1) = 5;
		data(1, 2) = 1;
		data(2, 0) = 1;
		data(2, 1) = 7;
		data(2, 2) = 1;

		data_set1.data = data.clone();
		data_set2.data = data.clone();
		data_set1.labels = labels1.clone();
		data_set2.labels = labels2.clone();

	}

	DataSet data_set1;
	DataSet data_set2;
};


#endif /* CLASSIFIERTEST_H_ */
