/*
 * HaarCascade.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef HAARCASCADE_H_
#define HAARCASCADE_H_


#include "Stage.h"

struct HaarCascade {
	int window_height;
	int window_width;
	std::vector<Stage> stages;
};


#endif
