/*
 * WebCamMedia.cpp
 *
 *  Created on: May 27, 2013
 *      Author: olehp
 */

#include "WebCamMedia.h"

#include <iostream>

using namespace std;


WebCamMedia::WebCamMedia(int w, int h, int web_cam_id) {

	capture = cvCaptureFromCAM(web_cam_id);
	if (!capture) {
		cerr << "No webcam found." << endl;
		return;
	}

	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, w);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, h);

	width = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
	height = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
}

void WebCamMedia::GetFrame(cv::Mat& frame) {
	frame = cvQueryFrame(capture);
}

WebCamMedia::~WebCamMedia() {
	cvReleaseCapture(&capture);
}

