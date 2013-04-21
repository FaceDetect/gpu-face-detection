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

	ASSERT_EQ(mat.rows, 1);
	ASSERT_EQ(mat.cols, 1);
	ASSERT_EQ(mat(0, 0), 100);
}

TEST(CreateMatTest, mat3x3) {
	Mat_<int> mat = CreateMatrix<int>(3, 3, []() { static int i = 0; return (++ i) * 2; });

	ASSERT_EQ(mat.rows, 3);
	ASSERT_EQ(mat.cols, 3);
	EXPECT_EQ(mat(0, 0), 2);
	EXPECT_EQ(mat(0, 1), 4);
	EXPECT_EQ(mat(1, 0), 8);
	EXPECT_EQ(sum(mat).val[0], 90);
}
