/*
 * feature_test.cc
 *
 *  Created on: Apr 20, 2013
 *      Author: olehp
 */

#include <gtest/gtest.h>
#include "MatTest.h"
#include "test_utils.h"

#include "Feature.h"
#include "constants.h"
#include "utils.h"

using namespace cv;
using namespace std;


class FeatureTest: public ::testing::Test {
public:
	virtual void SetUp() {

		r1 = Rectangle(3, 7, 14, 4, -1);
		r2 = Rectangle(3, 9, 14, 2, 2);
		r3 = Rectangle(8, 15, 10, 4, -1);
		r4 = Rectangle(13, 15, 5, 2, 2);
		r5 = Rectangle(8, 17, 5, 2, 2);


		f2rects = Feature(W_WIDTH + 1, r1, r2);
		f3rects = Feature(W_WIDTH + 1, r3, r4, r5);
	}

	Feature f2rects;
	Feature f3rects;

	Rectangle r1;
	Rectangle r2;
	Rectangle r3;
	Rectangle r4;
	Rectangle r5;
};

TEST_F(FeatureTest, RectsCoords) {
	EXPECT_EQ(3 + 7 * (W_WIDTH + 1), f2rects.rects_coords[0].p0);
	EXPECT_EQ(3 + 14 + 7 * (W_WIDTH + 1), f2rects.rects_coords[0].p1);
	EXPECT_EQ(3 + (7 + 4) * (W_WIDTH + 1), f2rects.rects_coords[0].p2);
	EXPECT_EQ(3 + 14 + (7 + 4) * (W_WIDTH + 1), f2rects.rects_coords[0].p3);
}

TEST_F(FeatureTest, FeatureEval) {
//	PrintMatrix(mat);

	Mat_<float> mat = CreateMatrix<float>(24, 24);
	Mat_<float> ii = mat.clone();
	ToIntegralImage(ii, SINGLE_SUM);

	float f2res = f2rects.Eval(ii);
	float f3res = f3rects.Eval(ii);
	float f2res_expect = 0;
	float f3res_expect = 0;


	for (int i = 0; i < HAAR_MAX_RECTS; i++) {

		int start_col = f2rects.rects[i].x;
		int end_col = f2rects.rects[i].x + f2rects.rects[i].w;
		int start_row = f2rects.rects[i].y;
		int end_row = f2rects.rects[i].y + f2rects.rects[i].h;


		if (f2rects.rects[i].wg != 0)
			f2res_expect += sum(mat.colRange(start_col, end_col).rowRange(start_row, end_row)).val[0] *
							f2rects.rects[i].wg;

		start_col = f3rects.rects[i].x;
		end_col = f3rects.rects[i].x + f3rects.rects[i].w;
		start_row = f3rects.rects[i].y;
		end_row = f3rects.rects[i].y + f3rects.rects[i].h;


		f3res_expect += sum(mat.colRange(start_col, end_col).rowRange(start_row, end_row)).val[0] *
						f3rects.rects[i].wg;
	}


	EXPECT_EQ(f2res_expect, f2res);
	EXPECT_EQ(f3res_expect, f3res);

}

TEST(GetFeatureSetTest, IsNotEmpty) {
	vector<Feature> set = GetFeatureSet();

	ASSERT_FALSE(set.empty());
}
