/*
 * ada_boost_test.cc
 *
 *  Created on: Apr 28, 2013
 *      Author: olehp
 */

#include <gtest/gtest.h>
#include "ClassifierTest.h"

#include "AdaBoost.h"

using namespace cv;


class AdaBoostTest : public ClassifierTest {

};
//
TEST_F(AdaBoostTest, Training) {
	AdaBoost ada_boost(data_set1);

	ada_boost.TrainWeak();

	EXPECT_EQ(1, ada_boost.GetStumps().size());
}

TEST_F(AdaBoostTest, Classification) {
	AdaBoost ada_boost(data_set1);
	ada_boost.TrainWeak();

	Mat_<label_t> labels1;
	ada_boost.Classify(data_set1.data, labels1);

	EXPECT_EQ(0, labels1(0, 0));
	EXPECT_EQ(0, labels1(1, 0));
	EXPECT_EQ(1, labels1(2, 0));
}
//
//TEST_F(AdaBoostTest, OverTraining) {
//	AdaBoost ada_boost(data_set1);
//	ada_boost.TrainWeak();
//	ada_boost.TrainWeak();
//
//	Mat_<label_t> labels1 = ada_boost.Classify(data_set1.data);
//
//	EXPECT_EQ(0, labels1(0, 0));
//	EXPECT_EQ(0, labels1(1, 0));
//	EXPECT_EQ(1, labels1(2, 0));
//}
