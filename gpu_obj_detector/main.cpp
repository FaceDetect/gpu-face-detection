/*
 * main.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */
#include <stdio.h>
#include "ObjectRecognizer.h"
#include <iostream>
#include "gpuDetectObjs.h"
#include "SubWindow.h"
#include "utils.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#define WEB_CAM_WIDTH 640
#define WEB_CAM_HEIGHT 480
#define W_NAME "Detection"

using namespace std;
using namespace cv;

void DetectAndDisplay(Mat img, const HaarCascade& haar_cascade, vector<SubWindow> subwindows);
void WebCamDetect(const HaarCascade& haar_cascade);
void ImgDetect(const HaarCascade& haar_cascade, const char *img_path);

int main(int argv, char **args)
{
	HaarCascade haar_cascade;
	LoadCascade("../../data/haarcascade_frontalface_alt.xml", haar_cascade);

	ImgDetect(haar_cascade, "../../data/lena.jpg");
//	WebCamDetect(haar_cascade);

	waitKey();
	return 0;

}

void ImgDetect(const HaarCascade& haar_cascade, const char *img_path) {
	Mat img = imread(img_path);

	if (img.empty()) {
		cerr << "Image " << img_path << " not found" << endl;
		return;
	}

	vector<SubWindow> subwindows;
	PrecalcSubwindows(img.cols, img.rows, haar_cascade.window_width, haar_cascade.window_height, subwindows);

	DetectAndDisplay(img, haar_cascade, subwindows);
}


void WebCamDetect(const HaarCascade& haar_cascade) {
	CvCapture *capture;

	capture = cvCaptureFromCAM(-1);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, WEB_CAM_WIDTH);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, WEB_CAM_HEIGHT);

	if (capture) {

		vector<SubWindow> subwindows;
		PrecalcSubwindows(WEB_CAM_WIDTH, WEB_CAM_HEIGHT, haar_cascade.window_width, haar_cascade.window_height, subwindows);

		Mat frame;

		while (true) {
			vector<SubWindow> objs = subwindows;

			frame = cvQueryFrame(capture);
			if (!frame.empty()) {
				DetectAndDisplay(frame, haar_cascade, objs);
			} else {
				cerr << "No frame" << endl;
				break;
			}
			int c = waitKey(10);
			if ((char)c == 'c') break;
		}
	}
}

void DetectAndDisplay(Mat img, const HaarCascade& haar_cascade, vector<SubWindow> subwindows) {
	Mat gray_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
	equalizeHist(gray_img, gray_img);
	gpuDetectObjs((Mat_<int>) gray_img, haar_cascade, subwindows);
	for (int i = 0; i < subwindows.size(); i++) {
		Point p1(subwindows[i].x, subwindows[i].y);
		Point p2(subwindows[i].x + subwindows[i].w, subwindows[i].y + subwindows[i].h);
		rectangle(img, p1, p2, Scalar(0, 0, 255));
	}
	imshow(W_NAME, img);
}
