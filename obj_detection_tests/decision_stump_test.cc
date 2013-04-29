/*
 * decision_stump_test.cc
 *
 *  Created on: Apr 26, 2013
 *      Author: olehp
 */

#include <gtest/gtest.h>
#include "MatTest.h"
#include "ClassifierTest.h"
#include "DecisionStump.h"
#include "DataSet.h"
#include "utils.h"

using namespace cv;

class DecisionStumpTest: public ClassifierTest {
public:
	virtual void SetUp() {
		MatTest::SetUp();
		ClassifierTest::SetUp();

		stump1 = DecisionStump(1, 10, true);
		stump2 = DecisionStump(1, 10, false);
	}

	DecisionStump stump1;
	DecisionStump stump2;
};

TEST_F(DecisionStumpTest, Classification) {
	Mat_<label_t> labels1 = stump1.Classify(mat3x3);

	EXPECT_EQ(0, labels1(0, 0));
	EXPECT_EQ(0, labels1(1, 0));
	EXPECT_EQ(1, labels1(2, 0));

	Mat_<label_t> labels2 = stump2.Classify(mat3x3);

	EXPECT_EQ(1, labels2(0, 0));
	EXPECT_EQ(1, labels2(1, 0));
	EXPECT_EQ(0, labels2(2, 0));
}

TEST_F(DecisionStumpTest, Training) {

	stump1.Train(data_set1, Mat_<float>::ones(3, 1));
	stump2.Train(data_set2, Mat_<float>::ones(3, 1));

	EXPECT_EQ(1, stump1.i_feature);
	EXPECT_EQ(5, stump1.threshold);
	EXPECT_TRUE(stump1.gt);


	EXPECT_EQ(1, stump2.i_feature);
	EXPECT_EQ(5, stump2.threshold);
	EXPECT_FALSE(stump2.gt);
}
