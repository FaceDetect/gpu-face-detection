/*
 * Stage.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef STAGE_H_
#define STAGE_H_

#define HAAR_MAX_TREES 213

#include "Tree.h"

struct Stage {
	Stage() :
		threshold(0),
		parent(0),
		next(0),
		valid(0) { }

	float threshold;
	int parent;
	int next;
	int valid;
	Tree trees[HAAR_MAX_TREES];
};


#endif /* STAGE_H_ */
