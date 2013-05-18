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
#include "SubWindow.h"

void gpuDetectObjs(cv::Mat_<int> img,
				   const HaarCascade& haar_cascade,
				   std::vector<SubWindow>& objs);
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
