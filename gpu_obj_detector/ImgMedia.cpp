/*
 * ImgMedia.cpp
 *
 *  Created on: May 27, 2013
 *      Author: olehp
 */

#include "ImgMedia.h"

using namespace cv;


ImgMedia::ImgMedia(const char *path)
{
	img = imread(path);
	width = img.cols;
	height = img.rows;
}

void ImgMedia::GetFrame(cv::Mat& frame) {
	frame = img.clone();
}

