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

int main(int argv, char **args)
{
//	cout << "sizeof(Stage): " << sizeof(Stage) << endl;

//
	HaarCascade haar_cascade;
	std::cout << "../../data/lena.jpg" << " ";
	Mat_<int> img = imread("../../data/lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//imshow("test", img);



	LoadCascade("../../data/haarcascade_frontalface_alt.xml", haar_cascade);
	gpuDetectObjs(img, haar_cascade);

//	ObjectRecognizer obj_rec;
//	std::cout << "../../data/lena.jpg" << " ";
//	obj_rec.LoadHaarCascade("../../data/haarcascade_frontalface_alt.xml");
//	obj_rec.LoadImage("../../data/lena.jpg");
//	obj_rec.Recognize();
//	obj_rec.UnloadImage();

}
