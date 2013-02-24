/*
 * utils.h
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include "Feature.h"

#define W_HEIGHT 24
#define W_WIDTH 24
#define HAAR_MAX_FEATURES

void GenerateFeatures(std::vector<Feature>& features);

#endif /* UTILS_H_ */
