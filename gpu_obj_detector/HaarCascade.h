/*
 * HaarCascade.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef HAARCASCADE_H_
#define HAARCASCADE_H_

#define HAAR_MAX_STAGES 22

#include "Stage.h"

struct HaarCascade {
	int window_height;
	int window_width;
	Stage stages[HAAR_MAX_STAGES];
};


#endif
