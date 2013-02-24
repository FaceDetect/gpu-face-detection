/*
 * Feature.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef FEATURE_H_
#define FEATURE_H_

#include <vector>
#include "Rectangle.h"

struct Feature {
	std::vector<Rectangle> rects;
	double tilted;
};


#endif /* FEATURE_H_ */
