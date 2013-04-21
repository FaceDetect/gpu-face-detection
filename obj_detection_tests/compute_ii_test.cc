/*
 * compute_ii_test.cc
 *
 *  Created on: Apr 20, 2013
 *      Author: olehp
 */

#include <gtest/gtest.h>
#include "MatTest.h"

#include "utils.h"

using namespace cv;

class ComputeIITest: public MatTest {
};

TEST_F(ComputeIITest, SingleSums) {

	Mat_<int> res1x1 = ComputeIntegralImage(mat1x1, SINGLE_SUM);

	ASSERT_EQ(res1x1.rows, mat1x1.rows);
	ASSERT_EQ(res1x1.cols, mat1x1.rows);
	ASSERT_EQ(res1x1(0, 0), mat1x1(0, 0));

	Mat_<int> res3x3 = ComputeIntegralImage(mat3x3, SINGLE_SUM);

	ASSERT_EQ(res3x3.rows, mat3x3.rows);
	ASSERT_EQ(res3x3.cols, mat3x3.rows);

	EXPECT_EQ(res3x3(0, 0), mat3x3(0, 0));
	EXPECT_EQ(res3x3(2, 2), sum(mat3x3).val[0]);
	EXPECT_EQ(res3x3(1, 2), sum(mat3x3.rowRange(0, 2)).val[0]);
	EXPECT_EQ(res3x3(2, 1), sum(mat3x3.colRange(0, 2)).val[0]);
	EXPECT_EQ(res3x3(1, 1), sum(mat3x3.colRange(0, 2).rowRange(0, 2)).val[0]);

}


TEST_F(ComputeIITest, SquaredSums) {

	Mat_<int> mat1x1sqr = mat1x1.clone();
	Mat_<int> mat3x3sqr = mat3x3.clone();

	for (auto &i : mat1x1sqr) i = SQR(i);
	for (auto &i : mat3x3sqr) i = SQR(i);

	Mat_<int> res1x1 = ComputeIntegralImage(mat1x1, SQUARED_SUM);

	ASSERT_EQ(res1x1.rows, mat1x1.rows);
	ASSERT_EQ(res1x1.cols, mat1x1.rows);
	ASSERT_EQ(res1x1(0, 0), mat1x1sqr(0, 0));

	Mat_<int> res3x3 = ComputeIntegralImage(mat3x3, SQUARED_SUM);

	ASSERT_EQ(res3x3.rows, mat3x3.rows);
	ASSERT_EQ(res3x3.cols, mat3x3.rows);

	EXPECT_EQ(res3x3(0, 0), mat3x3sqr(0, 0));
	EXPECT_EQ(res3x3(2, 2), sum(mat3x3sqr).val[0]);
	EXPECT_EQ(res3x3(1, 2), sum(mat3x3sqr.rowRange(0, 2)).val[0]);
	EXPECT_EQ(res3x3(2, 1), sum(mat3x3sqr.colRange(0, 2)).val[0]);
	EXPECT_EQ(res3x3(1, 1), sum(mat3x3sqr.colRange(0, 2).rowRange(0, 2)).val[0]);

}
