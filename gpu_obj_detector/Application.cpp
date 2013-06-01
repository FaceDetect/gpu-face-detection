/*
 * Application.cpp
 *
 *  Created on: May 27, 2013
 *      Author: olehp
 */

#include "Application.h"
#include "GpuObjDetector.h"
#include "CpuObjDetector.h"
#include "utils.h"

#include <iostream>
#include <opencv2/contrib/contrib.hpp>

using namespace cv;
using namespace std;


Application::Application(Media& source, const char *cascade_path, const char * w_name):
	media(source),
	window_name(w_name) {

	LoadCascade(cascade_path, haar_cascade);
}

void Application::Run() {
	Mat frame;

	GpuObjDetector gpu_det(media.GetWidth(), media.GetHeight(), haar_cascade);
	CpuObjDetector cpu_det(media.GetWidth(), media.GetHeight(), haar_cascade);
	detector = &gpu_det;

	while (true) {
		media.GetFrame(frame);

		if (frame.empty()) {
			cerr << "No frame" << endl;
			break;
		}


		TickMeter tm;
		tm.start();

		DetectAndDisplay(frame);

		tm.stop();

		float elapsed = tm.getTimeMilli();

		cout << "Elapsed: " << elapsed << endl;
		cout << "FPS: " << 1000 / elapsed << endl;



		char c = waitKey(10);
		if (c == 27) break;
		else if (c == 'c') detector = &cpu_det;
		else if (c == 'g') detector = &gpu_det;


	}
}

void Application::DetectAndDisplay(Mat& img) {
	Mat gray_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
//	equalizeHist(gray_img, gray_img);

	vector<Rectangle> objs;

	detector->Detect(((Mat_<int>) gray_img).ptr<int>(), objs);

	for (int i = 0; i < objs.size(); i++) {
		Point p1(objs[i].x, objs[i].y);
		Point p2(objs[i].x + objs[i].w, objs[i].y + objs[i].h);
		rectangle(img, p1, p2, Scalar(0, 0, 255));
	}

	imshow(window_name, img);
}

Application::~Application() {
	cvDestroyWindow(window_name);
}
