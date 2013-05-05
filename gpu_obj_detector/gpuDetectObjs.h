/*
 * gpuDetectObjs.h
 *
 *  Created on: May 4, 2013
 *      Author: olehp
 */

#ifndef GPUDETECTOBJS_H_
#define GPUDETECTOBJS_H_

#include <opencv2/opencv.hpp>
#include "HaarCascade.h"

void gpuDetectObjs(cv::Mat_<int> img, HaarCascade& haar_cascade);
bool gpuDetectObjsAt(int *ii,
					 int *ii2,
					 float scale,
					 int x,
					 int y,
					 int width,
					 int height,
					 int img_width,
					 int img_height,
					 HaarCascade& haar_cascade);



#endif /* GPUDETECTOBJS_H_ */
