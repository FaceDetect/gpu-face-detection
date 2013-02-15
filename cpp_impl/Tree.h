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
	double threshold;
	double left_val;
	double right_val;
	Feature feature;

};


#endif /* TREE_H_ */
