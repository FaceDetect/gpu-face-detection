/*
 * main.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include "TrainingData.h"
#include "utils.h"
#include "Feature.h"
#include <algorithm>
#include "DecisionStump.h"
#include "AdaBoost.h"

using namespace cv;
using namespace std;

int main( int argc, char **argv ){
//
	TrainingData td;
	td.LoadImages("test_pos.txt", "test_neg.txt");
	td.PrepareDataSet();

//	DecisionStump ds;
//	ds.gt = 1;
//	ds.threshold = 108125;
//	ds.i_feature = 792;
//
//	PrintMatrix(ds.Classify(td.data_set));


	//PrintMatrix((Mat_<int>)GET_MAT_COL(td.data_set, 738));

//	for (Feature &f : td.features) {
//		if (f.rects[1].wg > 3) {
//			f.PrintInfo();
//		}
//	}

//	td.features.at(136655).PrintInfo();
//
//	PrintMatrix(td.ii_pos.at(0));

	//PrintMatrix((Mat_<int>)GET_MAT_COL(td.data_set, 136655));

	AdaBoost ab(td);

	ab.Train(5);
	ENDL
	ENDL
	PrintMatrix(ab.Classify(td));
	ENDL
	ENDL


	TrainingData test_data;
	test_data.LoadImages("positives.txt", "negatives.txt");
	test_data.PrepareDataSet();

	ENDL
	PrintMatrix(ab.Classify(test_data));

//    Mat_<int> mat(5, 5);
//    Mat_<double> D(5, 1);
//    for (int row = 0; row < mat.rows; row++) {
//            for (int col = 0; col < mat.cols - 1; col++)
//                    mat(row, col) = ((row + col) * 10) % (row + 7);
//
//            mat(row, 4) = row % 2;
//
//            D(row, 0) = 1.0 / 5;
//    }
//
//    PrintMatrix(mat);
//
//    for(int &i : mat) {
//    	cout << i << " s";
//    }
//
//    DecisionStumpInfo ds = DecisionStump::Build(mat, D);
//
//	ds.ds.PrintInfo();
//	ENDL
//	PrintMatrix(ds.ds.Classify(mat));


    return 0;
}
