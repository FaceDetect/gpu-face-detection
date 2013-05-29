/*
 * ObjDetector.h
 *
 *  Created on: May 26, 2013
 *      Author: olehp
 */

#ifndef OBJDETECTOR_H_
#define OBJDETECTOR_H_

#include "HaarCascade.h"
#include <vector>

class ObjDetector {
public:
	virtual void Detect(const int *g_img, std::vector<Rectangle>& objs) = 0;
	virtual ~ObjDetector() { };
};

#endif /* OBJDETECTOR_H_ */
