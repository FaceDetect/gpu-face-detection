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
#include <algorithm>

#include "utils.h"
#include "constants.h"

using namespace std;
using namespace cv;


void TrainingData::LoadImages(const char* pos_image_list_path,
							  const char* neg_image_list_path) {

	ifstream pos_image_list(pos_image_list_path);
	ifstream neg_image_list(neg_image_list_path);

	string image_path;

	while(pos_image_list >> image_path)
		images_pos.push_back(imread(image_path, CV_LOAD_IMAGE_GRAYSCALE));

	while(neg_image_list >> image_path) {

		Mat img = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
		Mat res;
		resize(img, res, Size(W_WIDTH, W_HEIGHT));

		images_neg.push_back((Mat_<int>)res);
	}

	cout << "LOADING FINISHED." << endl;
}

TrainingData::TrainingData() {
	GenerateFeatures(features);
}



void TrainingData::PrepareDataSet() {

	ii_pos.resize(images_pos.size(), Mat_<int>(W_HEIGHT, W_WIDTH));
	ii_neg.resize(images_neg.size(), Mat_<int>(W_HEIGHT, W_WIDTH));

	transform(images_pos.begin(), images_pos.end(), ii_pos.begin(), ComputeIntegralImage);
	transform(images_neg.begin(), images_neg.end(), ii_neg.begin(), ComputeIntegralImage);

	data_set.create(images_pos.size() + images_neg.size(), features.size() + 1);


	int num_examples = 0;

	for (Mat_<int> &ii : ii_pos) {
		CreateDataEntry(ii, data_set.ptr<int>(num_examples), 1);
		num_examples++;
	}

	for (Mat_<int> &ii : ii_neg) {
		CreateDataEntry(ii, data_set.ptr<int>(num_examples), -1);
		num_examples++;
	}

	cout << "PREPARATION FINISHED." << endl;

}

void TrainingData::CreateDataEntry(Mat_<int> &ii, int *entry, int class_label) {

	int i = 0;
	for (Feature &f : features) {
		entry[i] = f.Eval(ii);
		i++;
	}

	entry[i] = class_label;
}

void TrainingData::ShowImages() {
	vector<Mat_<int> > vec;
	vec.insert(vec.end(), images_pos.begin(), images_pos.end());
	vec.insert(vec.end(), images_neg.begin(), images_neg.end());
	cout << vec.size() << endl;
	for (Mat_<int> m : vec) {
		namedWindow("Display window", CV_WINDOW_AUTOSIZE);
		imshow("Display window", (Mat_<uchar>)m);
		waitKey(0);
	}

}