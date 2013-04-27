/*
 * normalize_image_test.cc
 *
 *  Created on: Apr 21, 2013
 *      Author: olehp
 */

#include <gtest/gtest.h>
#include "MatTest.h"

#include "utils.h"

using namespace cv;


class NormalizeImageTest : public MatTest {
};

TEST_F(NormalizeImageTest, MeanTest) {
	EXPECT_EQ(mat1x1(0, 0), Mean(mat1x1));
	EXPECT_EQ(10, Mean(mat3x3));
}

TEST_F(NormalizeImageTest, StdDevTest) {
	EXPECT_EQ(0, StdDev(mat1x1));
	EXPECT_FLOAT_EQ(5.1639776, StdDev(mat3x3));
}

TEST_F(NormalizeImageTest, NormalizeMatrix) {
	Mat_<float> tmp1x1 = mat1x1.clone();
	Mat_<float> tmp3x3 = mat3x3.clone();

	NormalizeMat(tmp1x1);
	EXPECT_TRUE(MatrEq(tmp1x1, mat1x1));
}
