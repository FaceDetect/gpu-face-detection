/*
 * Rectangle.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef RECTANGLE_H_
#define RECTANGLE_H_

struct Rectangle {
	Rectangle() :
		x(0),
		y(0),
		w(0),
		h(0) { }

	Rectangle(int x, int y, int w, int h) :
		x(x),
		y(y),
		w(w),
		h(h) { }

	int x;
	int y;
	int w;
	int h;
};

struct WeightedRectangle : public Rectangle{
	WeightedRectangle() :
		Rectangle(),
		wg(0) { }

	int wg;
};

struct ScaledRectangle : public Rectangle {

	ScaledRectangle() :
		Rectangle(),
		scale(0) { }

	ScaledRectangle(int x, int y, int w, int h, float scale) :
		Rectangle(x, y, w, h),
		scale(scale) {
	}

	float scale;
};


#endif /* RECTANGLE_H_ */
