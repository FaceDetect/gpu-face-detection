/*
 * Feature.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef FEATURE_H_
#define FEATURE_H_

#include "Rectangle.h"

#define HAAR_MAX_RECTS 3

struct Feature {
	Rectangle rects[HAAR_MAX_RECTS];
	float tilted;
};


#endif /* FEATURE_H_ */
