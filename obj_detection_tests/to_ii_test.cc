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

class ToIITest: public MatTest {
};

TEST_F(ToIITest, SingleSums) {


	Mat_<int> res1x1 = mat1x1.clone();
	ToIntegralImage(res1x1, SINGLE_SUM);

	ASSERT_EQ(mat1x1.rows + 1, res1x1.rows);
	ASSERT_EQ(mat1x1.rows + 1, res1x1.cols);
	ASSERT_EQ(mat1x1(0, 0), res1x1(1, 1));

	Mat_<int> res3x3 = mat3x3.clone();
	ToIntegralImage(res3x3, SINGLE_SUM);

	ASSERT_EQ(mat3x3.rows + 1, res3x3.rows);
	ASSERT_EQ(mat3x3.rows + 1, res3x3.cols);

	EXPECT_EQ(mat3x3(0, 0), res3x3(1, 1));
	EXPECT_EQ(sum(mat3x3).val[0], res3x3(3, 3));
	EXPECT_EQ(sum(mat3x3.rowRange(0, 2)).val[0], res3x3(2, 3));
	EXPECT_EQ(sum(mat3x3.colRange(0, 2)).val[0], res3x3(3, 2));
	EXPECT_EQ(sum(mat3x3.colRange(0, 2).rowRange(0, 2)).val[0], res3x3(2, 2));

}


TEST_F(ToIITest, SquaredSums) {

	Mat_<int> mat1x1sqr = mat1x1.clone();
	Mat_<int> mat3x3sqr = mat3x3.clone();

	for (auto &i : mat1x1sqr) i = sqr(i);
	for (auto &i : mat3x3sqr) i = sqr(i);

	Mat_<int> res1x1 = mat1x1.clone();
	ToIntegralImage(res1x1, SQUARED_SUM);

	ASSERT_EQ(mat1x1.rows + 1, res1x1.rows);
	ASSERT_EQ(mat1x1.rows + 1, res1x1.cols);
	ASSERT_EQ(mat1x1sqr(0, 0), res1x1(1, 1));

	Mat_<int> res3x3 = mat3x3.clone();
	ToIntegralImage(res3x3, SQUARED_SUM);

	ASSERT_EQ(mat3x3.rows + 1, res3x3.rows);
	ASSERT_EQ(mat3x3.rows + 1, res3x3.cols);

	EXPECT_EQ(mat3x3sqr(0, 0), res3x3(1, 1));
	EXPECT_EQ(sum(mat3x3sqr).val[0], res3x3(3, 3));
	EXPECT_EQ(sum(mat3x3sqr.rowRange(0, 2)).val[0], res3x3(2, 3));
	EXPECT_EQ(sum(mat3x3sqr.colRange(0, 2)).val[0], res3x3(3, 2));
	EXPECT_EQ(sum(mat3x3sqr.colRange(0, 2).rowRange(0, 2)).val[0], res3x3(2, 2));

}
