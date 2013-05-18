/*
 * main.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#include <stdio.h>
#include "ObjectRecognizer.h"
#include <iostream>
#include "utils.h"
#include "gpuDetectObjs.h"
#include "Stage.h"

using namespace std;
using namespace cv;

//#define HANDLE_ERROR(x) x


int main(int argv, char **args)
{
//	cout << "sizeof(Stage): " << sizeof(Stage) << endl;


	HaarCascade haar_cascade;
	std::cout << "../../data/lena.jpg" << " ";
	Mat_<int> img = imread("../../data/lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	LoadCascade("../../data/haarcascade_frontalface_alt.xml", haar_cascade);

	gpuDetectObjs(img, haar_cascade);



//	Mat_<int> test(3, 3);
//
//	for (int i = 0; i < 3; i++) {
//		for (int j = 0; j < 3; j++) {
//			test(i, j) = i + 2 * (j + 1);
//			cout << test(i, j) << "\t";
//		}
//		cout << endl;
//	}
//
//	int *dev_ii;
//	int *dev_ii2;
//	int *ii = new int[4 * 4];
//	gpuComputeII(test.ptr<int>(), &dev_ii, &dev_ii2, 3, 3);
//
//	cudaMemcpy(ii, dev_ii, sizeof(int) * 4 * 4, cudaMemcpyDeviceToHost);
//
//	Mat_<int> res(4, 4, ii);
//
//	cout << endl;
//	for (int i = 0; i < 4; i++) {
//		for (int j = 0; j < 4; j++) {
//			cout << res(i, j) << "\t";
//		}
//		cout << endl;
//	}



//	ObjectRecognizer obj_rec;
//	std::cout << "../../data/lena.jpg" << " ";
//	obj_rec.LoadHaarCascade("../../data/haarcascade_frontalface_alt.xml");
//	obj_rec.LoadImage("../../data/lena.jpg");
//	obj_rec.Recognize();
//	obj_rec.UnloadImage();

}
