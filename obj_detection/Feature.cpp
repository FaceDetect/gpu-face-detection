/*
 * Feature.cpp
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#include "Feature.h"
#include "Rect.h"

#include <iostream>

using namespace std;


Feature::Feature() {
}

Feature::Feature(int offset, Rectangle r1, Rectangle r2, Rectangle r3) {
	rects[0] = r1;
	rects[1] = r2;
	rects[2] = r3;

	for (int i = 0; i < HAAR_MAX_FEATURES; i++) {

		if (rects[i].wg == 0) continue;

		rects_coords[i].p0 = rects[i].x + rects[i].y * offset;
		rects_coords[i].p1 = rects[i].x + rects[i].w + rects[i].y * offset;
		rects_coords[i].p2 = rects[i].x + (rects[i].y + rects[i].h) * offset;
		rects_coords[i].p3 = (rects[i].x + rects[i].w) + (rects[i].y + rects[i].h) * offset;
	}
}

Feature::Feature(int offset,
		 	 	 int x0, int y0, int w0, int h0, int wg0,
				 int x1, int y1, int w1, int h1, int wg1,
				 int x2, int y2, int w2, int h2, int wg2) :
	Feature(offset, Rectangle(x0, y0, w0, h0, wg0),
					Rectangle(x1, y1, w1, h1, wg1),
					Rectangle(x2, y2, w2, h2, wg2)) {

}

float Feature::Eval(const cv::Mat_<float> &ii) const {
	float sum = 0;
	const float * data = ii.ptr<float>(0);

	for (int i = 0; i < HAAR_MAX_FEATURES; i++) {
		if (rects[i].wg == 0) continue;
		sum += ((data[rects_coords[i].p0] + data[rects_coords[i].p3] - data[rects_coords[i].p2] - data[rects_coords[i].p1]) * rects[i].wg);
	}

	return sum;
}

void Feature::PrintInfo() {

	cout << "********FEATURE INFO********" << endl;
	for (int i = 0; i < HAAR_MAX_FEATURES; i++) {
		if (rects[i].wg == 0) continue;
		cout << "x" << i << ": " << rects[i].x << endl;
		cout << "y" << i << ": " << rects[i].y << endl;
		cout << "w" << i << ": " << rects[i].w << endl;
		cout << "h" << i << ": " << rects[i].h << endl;
		cout << "wg" << i << ": " << rects[i].wg << endl;
	}

	cout << "****************************" << endl;

}
