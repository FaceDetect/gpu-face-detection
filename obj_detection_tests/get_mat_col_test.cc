/*
 * get_mat_col_test.cc
 *
 *  Created on: Apr 19, 2013
 *      Author: olehp
 */

#include <gtest/gtest.h>
#include "MatTest.h"

#include "utils.h"

using namespace cv;


class GetMatColTest : public MatTest {
};

TEST_F(GetMatColTest, FirstCol) {
	Mat_<int> first_col = GET_MAT_COL(mat3x3, 0);

	ASSERT_EQ(first_col.rows, mat3x3.rows);
	ASSERT_EQ(first_col.cols, 1);
	EXPECT_EQ(first_col(0, 0), mat3x3(0, 0));
	EXPECT_EQ(first_col(1, 0), mat3x3(1, 0));
	EXPECT_EQ(first_col(2, 0), mat3x3(2, 0));
}

TEST_F(GetMatColTest, SecondCol) {
	Mat_<int> second_col = GET_MAT_COL(mat3x3, 1);

	ASSERT_EQ(second_col.rows, mat3x3.rows);
	ASSERT_EQ(second_col.cols, 1);
	EXPECT_EQ(second_col(0, 0), mat3x3(0, 1));
	EXPECT_EQ(second_col(1, 0), mat3x3(1, 1));
	EXPECT_EQ(second_col(2, 0), mat3x3(2, 1));
}

TEST_F(GetMatColTest, ThirdCol) {
	Mat_<int> third_col = GET_MAT_COL(mat3x3, 2);

	ASSERT_EQ(third_col.rows, mat3x3.rows);
	ASSERT_EQ(third_col.cols, 1);
	EXPECT_EQ(third_col(0, 0), mat3x3(0, 2));
	EXPECT_EQ(third_col(1, 0), mat3x3(1, 2));
	EXPECT_EQ(third_col(2, 0), mat3x3(2, 2));
}
