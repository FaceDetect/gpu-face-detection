/*
 * main.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */
#include <stdio.h>
#include "CpuObjDetector.h"
#include <iostream>
#include "SubWindow.h"
#include "utils.h"
#include <time.h>
#include "GpuObjDetector.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#define WEB_CAM_WIDTH 640
#define WEB_CAM_HEIGHT 480
#define W_NAME "Detection"

using namespace std;
using namespace cv;

void DetectAndDisplay(Mat& img, ObjDetector& detector);
void WebCamDetect(ObjDetector& detector);
void ImgDetect(ObjDetector& detector, const char *img_path);

int main(int argv, char **args)
{
	HaarCascade haar_cascade;
	LoadCascade("../../data/haarcascade_frontalface_alt.xml", haar_cascade);

//	GpuObjDetector detector(WEB_CAM_WIDTH, WEB_CAM_HEIGHT, haar_cascade);
	GpuObjDetector detector(600, 597, haar_cascade);

	ImgDetect(detector, "../../data/judybats.jpg");

//	WebCamDetect(detector);

	waitKey();
	cvDestroyWindow(W_NAME);
	return 0;

}

void ImgDetect(ObjDetector& detector, const char *img_path) {
	Mat img = imread(img_path);

	if (img.empty()) {
		cerr << "Image " << img_path << " not found" << endl;
		return;
	}

	DetectAndDisplay(img, detector);
}


void WebCamDetect(ObjDetector& detector) {
	CvCapture *capture;

	capture = cvCaptureFromCAM(-1);
	if (!capture) {
		cerr << "No webcam found." << endl;
		return;
	}

	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, WEB_CAM_WIDTH);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, WEB_CAM_HEIGHT);


	Mat frame;

	float fps;


	while (true) {
		frame = cvQueryFrame(capture);
		if (!frame.empty()) {

			const clock_t begin_time = clock();

			DetectAndDisplay(frame, detector);

			float elapsed = float( clock () - begin_time ) /  CLOCKS_PER_SEC;

			fps = 1.0 / elapsed;

			cout << "FPS: " << fps << endl;



		} else {
			cerr << "No frame" << endl;
			break;
		}
		int c = waitKey(10);
		if ((char)c == 'c') break;
	}

	cvReleaseCapture(&capture);

}

void DetectAndDisplay(Mat& img, ObjDetector& detector) {
	Mat gray_img = img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
//	equalizeHist(gray_img, gray_img);

	vector<SubWindow> objs;

	detector.Detect(((Mat_<int>) gray_img).ptr<int>(), objs);

	for (int i = 0; i < objs.size(); i++) {
		Point p1(objs[i].x, objs[i].y);
		Point p2(objs[i].x + objs[i].w, objs[i].y + objs[i].h);
		rectangle(img, p1, p2, Scalar(0, 0, 255));
	}
	imshow(W_NAME, img);
}
