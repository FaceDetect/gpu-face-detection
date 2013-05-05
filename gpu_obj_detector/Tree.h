/*
 * Tree.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef TREE_H_
#define TREE_H_

#include "Feature.h"

struct Tree {
	Tree() :
		threshold(0),
		left_val(0),
		right_val(0),
		valid(0) { }
	float threshold;
	float left_val;
	float right_val;
	int valid;
	Feature feature;

};


#endif /* TREE_H_ */
