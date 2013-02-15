/*
 * Stage.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef STAGE_H_
#define STAGE_H_

#include "Tree.h"

struct Stage {
	double threshold;
	int parent;
	int next;
	std::vector<Tree> trees;
};


#endif /* STAGE_H_ */
