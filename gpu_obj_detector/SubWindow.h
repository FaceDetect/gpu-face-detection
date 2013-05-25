/*
 * SubWindow.h
 *
 *  Created on: May 6, 2013
 *      Author: olehp
 */

#ifndef SUBWINDOW_H_
#define SUBWINDOW_H_

struct SubWindow {
	SubWindow(int x, int y, int w, int h, float scale) :
		x(x),
		y(y),
		w(w),
		h(h),
		scale(scale),
		is_object(1) { }

	int x;
	int y;
	int w;
	int h;
	int is_object;
	float scale;
};


#endif /* SUBWINDOW_H_ */
