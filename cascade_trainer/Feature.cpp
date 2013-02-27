/*
 * Feature.cpp
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#include "Feature.h"
#include "Rect.h"

Feature::Feature() {
}

Feature::Feature(int offset,
		 	 	 int x0, int y0, int w0, int h0, int wg0,
				 int x1, int y1, int w1, int h1, int wg1,
				 int x2, int y2, int w2, int h2, int wg2) {

	rects[0].x = x0;
	rects[0].y = y0;
	rects[0].w = w0;
	rects[0].h = h0;
	rects[0].wg = wg0;

	rects[1].x = x1;
	rects[1].y = y1;
	rects[1].w = w1;
	rects[1].h = h1;
	rects[2].wg = wg1;

	rects[2].x = x2;
	rects[2].y = y2;
	rects[2].w = w2;
	rects[2].h = h2;
	rects[2].wg = wg2;

	for (int i = 0; i < HAAR_MAX_FEATURES; i++) {
		rects_coords[i].p0 = rects[i].x + rects[i].y;
		rects_coords[i].p1 = rects[i].x + (rects[i].y * offset);
		rects_coords[i].p2 = rects[i].x + (rects[i].y + rects[i].h) * offset;
		rects_coords[i].p3 = (rects[i].x + rects[i].w) + (rects[i].y + rects[i].h) * offset;
	}
}

int Feature::Eval(const cv::Mat_<int> &ii) const {
	int sum = 0;
	const int * data = ii.ptr<int>(0);

	for (int i = 0; i < HAAR_MAX_FEATURES; i++) {
		if (rects[i].wg == 0) continue;
		sum += data[rects_coords[i].p0] + data[rects_coords[i].p3] - data[rects_coords[i].p2] - data[rects_coords[i].p1];
	}

	return sum;
}
