/*
 * Rectangle.h
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#ifndef RECTANGLE_H_
#define RECTANGLE_H_

struct Rectangle {

	Rectangle() :
		x(0),
		y(0),
		w(0),
		h(0),
		wg(0) {	}

	Rectangle(int x, int y,	int w, int h, int wg) :
		x(x),
		y(y),
		w(w),
		h(h),
		wg(wg) { }

	int x;
	int y;
	int w;
	int h;
	int wg;
};


#endif /* RECTANGLE_H_ */
