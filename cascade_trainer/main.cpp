/*
 * main.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "TrainingData.h"
#include "utils.h"
#include "Feature.h"

using namespace cv;
using namespace std;

int main( int argc, char **argv ){

	Mat_<int> image;

    image = imread("../data/lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    const int * ptr = image.ptr<int>(0);
	cout << image.rows << endl;
	cout << image.cols << endl;
	cout << (int)ptr[514] << endl;
	cout << (int)ptr[515] << endl;
	cout << image(-1, -1) << endl;
	cout << image(-1, -1) << endl;


    if(!image.data) {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow("Display window", CV_WINDOW_AUTOSIZE);
    imshow("Display window", (Mat_<uchar>)image);

    waitKey(0);

//	TrainingData td;
//	td.LoadImages("positives.txt");

//	vector<Feature> feats;
//
//	GenerateFeatures(feats);
//
//	cout << feats.size() << endl;

    return 0;
}
