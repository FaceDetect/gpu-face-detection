/*
 * CpuObjDetector.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#include "CpuObjDetector.h"
#include "utils.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

using namespace std;

CpuObjDetector::CpuObjDetector(int w, int h, HaarCascade& cascade) :
		pic_width(w),
		pic_height(h),
		haar_cascade(cascade) {

	grayscaled_bytes = new int[pic_width * pic_height];
	ii = new int[(pic_width + 1) * (pic_height + 1)];
	ii2 = new int[(pic_width + 1) * (pic_height + 1)];

}

CpuObjDetector::~CpuObjDetector() {
	delete [] grayscaled_bytes;
	delete [] ii;
	delete [] ii2;
}

void CpuObjDetector::Detect(const int *g_img, vector<Rectangle>& objs) {
	memcpy(grayscaled_bytes, g_img, pic_width * pic_height * sizeof(int));

	ComputeIntegralImages();

	double scale = 1.0;

	int width = haar_cascade.window_width;
	int height = haar_cascade.window_height;

	while (OR_MIN(width, height) <= OR_MIN(pic_width, pic_height)) {

		int x_step = 5;
		int y_step = 5;

		double inv = 1.0 / (width * height);

		for (int y = 0; y < pic_height - height; y += y_step) {
			for (int x = 0; x < pic_width - width; x += x_step) {

				double mean = RectSum(ii, x, y, width, height) * inv;
				double variance = RectSum(ii2, x, y, width, height) * inv - OR_SQR(mean);
				double std_dev = 1;

				if (variance >= 0)
					std_dev = sqrt(variance);

				if (std_dev < 10)
					continue;

				if (StagesPass(x, y, scale, inv, std_dev)) {
					objs.push_back(Rectangle(x, y, width, height));
				}
			}
		}

		scale = scale * 1.2;
		width = (int)(haar_cascade.window_width * scale);
		height = (int)(haar_cascade.window_height * scale);
	}
}

void CpuObjDetector::ComputeIntegralImages() {
	memset((void *)ii, 0, sizeof(int) * (pic_width + 1) * (pic_height + 1));
	memset((void *)ii2, 0, sizeof(int) * (pic_width + 1) * (pic_height + 1));
	ComputeIIs(grayscaled_bytes, ii, ii2, pic_width, pic_height);
}

bool CpuObjDetector::StagesPass(int x, int y, double scale, double inv, double std_dev) {

	for (int i = 0; i < HAAR_MAX_STAGES; i++) {
		Stage &stage = haar_cascade.stages[i];
		if (!stage.valid) break;

		double tree_sum = 0;//TreesPass(stage, x, y, scale, inv, std_dev);


	    for (int j = 0; j < HAAR_MAX_TREES; j++) {
	    	Tree& tree = stage.trees[j];
	    	if (!tree.valid) break;

	        double rects_sum = 0;//RectsPass(tree, x, y, scale) * inv;

	        for (int k = 0; k < HAAR_MAX_RECTS; k++) {
				WeightedRectangle &rect = tree.feature.rects[k];
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


inline int CpuObjDetector::RectSum(int* ii, int x, int y, int w, int h) {

	return MatrVal(ii, y, x, pic_width + 1) +
		   MatrVal(ii, y + h, x + w, pic_width + 1) -
		   MatrVal(ii, y, x + w, pic_width + 1) -
		   MatrVal(ii, y + h, x, pic_width + 1);
}
