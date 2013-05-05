/*
 * ObjectRecognizer.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#include "ObjectRecognizer.h"
#include "utils.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "gpu_compute_ii.h"
#include "gpuDetectObjs.h"

using namespace std;
using namespace cv;

ObjectRecognizer::ObjectRecognizer() :
		pic_width(-1),
		pic_height(-1),
		grayscaled_pic(0, 0),
		grayscaled_bytes(NULL),
		ii(NULL),
		ii2(NULL) {
}

ObjectRecognizer::~ObjectRecognizer() {
}

void ObjectRecognizer::LoadHaarCascade(const char *path) {

	LoadCascade(path, haar_cascade);
}

void ObjectRecognizer::Recognize() {
//	const clock_t begin_time = clock();

	ComputeIntegralImages();

//	std::cout << endl << "Time elapsed: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

//	double scale = 1.0;

	int width = 123;//haar_cascade.window_width;
	int height = 123;//haar_cascade.window_height;

	int x = 244;
	int y = 216;
	float scale = 6.19174248;
//	while (OR_MIN(width, height) <= OR_MIN(pic_width, pic_height)) {
//
//		int x_step = OR_MAX(1, OR_MIN(4, floor(width / 10)));
//		int y_step = OR_MAX(1, OR_MIN(4, floor(height / 10)));
//
		double inv = 1.0 / (width * height);
//
//		for (int y = 0; y < pic_height - height; y += y_step) {
//			for (int x = 0; x < pic_width - width; x += x_step) {

//				cout << "X: " << x << endl;
//				cout << "Y: " << y << endl << endl;;


				double mean = RectSum(ii, x, y, width, height) * inv;
				double variance = RectSum(ii2, x, y, width, height) * inv - OR_SQR(mean);



				double std_dev = 1;

				if (variance >= 0)
					std_dev = sqrt(variance);


//				if (std_dev < 10)
//					continue;

				cout << "CPU mean: " << mean << endl;
				cout << "CPU variance: " << variance << endl;
				cout << "CPU std_dev: " << std_dev << endl;

//				if (gpuDetectObjsAt(ii, ii2, 6.19174248, 244, 216, 123, 123, pic_width, pic_height, haar_cascade)) {
				if (StagesPass(x, y, scale, inv, std_dev)) {
					cout << x << " " << y << " " << width << " " << height << " ";
				}
//			}
//		}
//
//		scale = scale * 1.2;
//		width = (int)(haar_cascade.window_width * scale);
//		height = (int)(haar_cascade.window_height * scale);
//	}
}

void ObjectRecognizer::LoadImage(const char* path) {

	grayscaled_pic = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

	pic_width = grayscaled_pic.cols;
	pic_height = grayscaled_pic.rows;
	grayscaled_bytes = grayscaled_pic.ptr<int>();

	ii = new int[pic_width * pic_height];
	ii2 = new int[pic_width * pic_height];
}

void ObjectRecognizer::UnloadImage() {
	delete [] ii;
	delete [] ii2;
}

void ObjectRecognizer::ComputeIntegralImages() {

//	gpuComputeII(grayscaled_bytes, ii, ii2, pic_height, pic_width);
	for (int y = 0; y < pic_height; y++) {
		for (int x = 0; x < pic_width; x++) {
			SetMatrVal(ii, y, x,
					   MatrVal(grayscaled_bytes, y, x) -
					   MatrVal(ii, y - 1, x - 1) +
					   MatrVal(ii, y, x - 1) +
					   MatrVal(ii, y - 1, x));

			SetMatrVal(ii2, y, x,
					   MatrVal(grayscaled_bytes, y, x) * MatrVal(grayscaled_bytes, y, x) -
					   MatrVal(ii2, y - 1, x - 1) +
					   MatrVal(ii2, y, x - 1) +
					   MatrVal(ii2, y - 1, x));
		}
	}
}

bool ObjectRecognizer::StagesPass(int x, int y, double scale, double inv, double std_dev) {

	for (int i = 0; i < HAAR_MAX_STAGES; i++) {
		Stage &stage = haar_cascade.stages[i];
		if (!stage.valid) break;

		double tree_sum = 0;//TreesPass(stage, x, y, scale, inv, std_dev);


	    for (int j = 0; j < HAAR_MAX_TREES; j++) {
	    	Tree& tree = stage.trees[j];
	    	if (!tree.valid) break;

	        double rects_sum = 0;//RectsPass(tree, x, y, scale) * inv;

	        for (int k = 0; k < HAAR_MAX_RECTS; k++) {
				Rectangle &rect = tree.feature.rects[k];
				if (rect.wg == 0) break;

				rects_sum = rects_sum + RectSum(ii, x + (int)(rect.x * scale),
													y + (int)(rect.y * scale),
													(int)(rect.w * scale),
													(int)(rect.h * scale)) * rect.wg;
			}

	        tree_sum += ((rects_sum * inv < tree.threshold * std_dev) ? tree.left_val : tree.right_val);
	    }



		if (tree_sum < stage.threshold) {
			return false;
		}
	}

	return true;
}

double ObjectRecognizer::TreesPass(Stage &stage, int x, int y, double scale, double inv, double std_dev) {

    double tree_sum = 0;

    for (int i = 0; i < HAAR_MAX_TREES; i++) {
    	Tree& tree = stage.trees[i];
    	if (!tree.valid) break;

        double rects_sum = 0;//RectsPass(tree, x, y, scale) * inv;

        for (int k = 0; k < HAAR_MAX_RECTS; k++) {
			Rectangle &rect = tree.feature.rects[k];
			if (rect.wg == 0) break;

			rects_sum = rects_sum + RectSum(ii, x + (int)(rect.x * scale),
												y + (int)(rect.y * scale),
												(int)(rect.w * scale),
												(int)(rect.h * scale)) * rect.wg;
		}

        tree_sum += ((rects_sum * inv < tree.threshold * std_dev) ? tree.left_val : tree.right_val);
    }

    return tree_sum;
}

double ObjectRecognizer::RectsPass(Tree &tree, int x, int y, double scale) {
	double rects_sum = 0;
	for (int i = 0; i < HAAR_MAX_RECTS; i++) {
		Rectangle &rect = tree.feature.rects[i];
		if (rect.wg == 0) break;

		rects_sum = rects_sum + RectSum(ii, x + (int)(rect.x * scale),
											y + (int)(rect.y * scale),
											(int)(rect.w * scale),
											(int)(rect.h * scale)) * rect.wg;
	}

	return rects_sum;
}


inline int ObjectRecognizer::RectSum(int* ii, int x, int y, int w, int h) {

	return MatrVal(ii, y - 1, x - 1) +
		   MatrVal(ii, y + h - 1, x + w - 1) -
		   MatrVal(ii, y - 1, x + w - 1) -
		   MatrVal(ii, y + h - 1, x - 1);
//
//	return MatrVal(ii, y, x) +
//		   MatrVal(ii, y + w, x + h) -
//		   MatrVal(ii, y, x + w) -
//		   MatrVal(ii, y + h, x);
}
