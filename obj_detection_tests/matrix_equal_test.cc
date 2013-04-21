/*
 * matrix_equal_test.cc
 *
 *  Created on: Apr 20, 2013
 *      Author: olehp
 */

#include "MatTest.h"

#include "utils.h"

using namespace cv;

class MatrixEqualTest : public MatTest {
};

TEST_F(MatrixEqualTest, equal) {
	Mat_<int> that_mat = mat3x3.clone();

	ASSERT_TRUE(MatrEq(mat3x3, that_mat));
}
