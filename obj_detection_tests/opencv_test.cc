/*
 * opencv_test.cc
 *
 *  Created on: Apr 20, 2013
 *      Author: olehp
 */


#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

using namespace cv;


// !!! Copy constructors and assignment operators: only references are copied, not the data. To perform a deep copy use mat.clone()




TEST(DISABLED_MatTests, CopyConstr) {
	Mat_<int> mat(3, 3);
	int i = 0;
	for (int &iter : mat) iter = ++i;

	Mat_<int> mat_other = mat;

	mat_other(0, 0) = -1;

	ASSERT_NE(mat(0, 0), mat_other(0, 0));
}


TEST(DISABLED_MatTests, Assignment) {
	Mat_<int> mat(3, 3);
	int i = 0;
	for (int &iter : mat) iter = ++i;

	Mat_<int> mat_other(3, 3);

	mat_other = mat;

	mat_other(0, 0) = -1;

	ASSERT_NE(mat(0, 0), mat_other(0, 0));
}
