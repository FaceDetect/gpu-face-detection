/*
 * test_create_mat.cc
 *
 *  Created on: Apr 20, 2013
 *      Author: olehp
 */

#include <gtest/gtest.h>

#include "test_utils.h"

using namespace cv;

TEST(CreateMatTest, mat1x1) {
	Mat_<int> mat = CreateMatrix<int>(1, 1, []() { return 100; });

	ASSERT_EQ(1, mat.rows);
	ASSERT_EQ(1, mat.cols);
	ASSERT_EQ(100, mat(0, 0));
}

TEST(CreateMatTest, mat3x3) {
	Mat_<int> mat = CreateMatrix<int>(3, 3, []() { static int i = 0; return (++ i) * 2; });

	ASSERT_EQ(3, mat.rows);
	ASSERT_EQ(3, mat.cols);
	EXPECT_EQ(2, mat(0, 0));
	EXPECT_EQ(4, mat(0, 1));
	EXPECT_EQ(8, mat(1, 0));
	EXPECT_EQ(90, sum(mat).val[0]);
}
