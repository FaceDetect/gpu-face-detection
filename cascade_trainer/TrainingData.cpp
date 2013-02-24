/*
 * TrainingData.cpp
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#include "TrainingData.h"

#include <fstream>
#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;


void TrainingData::LoadImages(const char* image_list_path) {
	ifstream image_list(image_list_path);
	string image_path;

	while(image_list >> image_path) {
		images.push_back(imread(image_path, CV_LOAD_IMAGE_GRAYSCALE));
	}

	cout << "Done." << endl;
}
