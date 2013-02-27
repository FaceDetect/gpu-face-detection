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

using namespace cv;
using namespace std;

int main( int argc, char **argv ){

//	Mat_<int> image;
//	Mat res;
//    image = imread("../data/lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//
//    resize(image, res, Size(100, 100));
//    const int * ptr = image.ptr<int>(0);
////	cout << image.rows << endl;
////	cout << image.cols << endl;
////	cout << (int)ptr[514] << endl;
////	cout << (int)ptr[515] << endl;
////	cout << image(-1, -1) << endl;
////	cout << image(-1, -1) << endl;
//
//
//    if(!image.data) {
//        cout <<  "Could not open or find the image" << std::endl ;
//        return -1;
//    }
//
//    namedWindow("Display window", CV_WINDOW_AUTOSIZE);
//    imshow("Display window", (Mat_<uchar>)res);
//
//    waitKey(0);

	//TrainingData td;
	//td.LoadImages("positives.txt", "negatives.txt");
	//td.ShowImages();
	//td.PrepareDataSet();

	Mat_<int> mat(5, 5);
	Mat_<double> D(5, 1);
	for (int row = 0; row < mat.rows; row++) {
		for (int col = 0; col < mat.cols - 1; col++)
			mat(row, col) = ((row + col) * 10) % (row + 7);

		mat(row, 4) = row % 2;

		D(row, 0) = 1.0 / 5;
	}



	PrintMatrix(mat);

//	ENDL
//
//	PrintMatrix(mat.colRange(4, 5));

	DecisionStump ds = DecisionStump::Build(mat, D);

	ds.PrintInfo();
	ENDL
	PrintMatrix(ds.Classify(mat));

//	vector<Feature> feats;
//
//	GenerateFeatures(feats);
//
//	cout << feats.size() << endl;
//
//    return 0;
}
