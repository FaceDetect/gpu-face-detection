/*
 * SubWindow.h
 *
 *  Created on: May 6, 2013
 *      Author: olehp
 */

#ifndef SUBWINDOW_H_
#define SUBWINDOW_H_

struct SubWindow {

	SubWindow() :
		x(0),
		y(0),
		w(0),
		h(0),
		scale(0) { }

	SubWindow(int x, int y, int w, int h, float scale) :
		x(x),
		y(y),
		w(w),
		h(h),
		scale(scale) { }

	int x;
	int y;
	int w;
	int h;
	float scale;
};


#endif /* SUBWINDOW_H_ */
