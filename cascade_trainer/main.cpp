/*
 * main.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#include <opencv2/opencv.hpp>
#include <iostream>
//#include "DataSet.h"
//#include "utils.h"
//#include "Feature.h"
//#include <algorithm>
//#include "DecisionStump.h"
//#include "AdaBoost.h"

#include "gpu_compute_ii.h"
#include "utils.h"

using namespace cv;
using namespace std;

int main( int argc, char **argv ){

	Mat_<int> mat(3, 3);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			mat(i, j) = i + 2 * j;
			cout << mat(i, j) << "\t";
		}
		cout << endl;
	}

	int res[9], sq_res[9];
//	const int * data = mat.ptr<int>;
	gpuComputeII(mat.ptr<int>(), res, sq_res, 3, 3);

	Mat_<int> res_mat(3, 3, sq_res);
	ENDL
	ENDL
	PrintMatrix(res_mat);
	ENDL
	ENDL
	ToIntegralImage(mat, SQUARED_SUM);
	PrintMatrix(mat);
    return 0;
}
