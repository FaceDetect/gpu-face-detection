/*
 * ObjDetector.h
 *
 *  Created on: May 26, 2013
 *      Author: olehp
 */

#ifndef OBJDETECTOR_H_
#define OBJDETECTOR_H_

#include "HaarCascade.h"
#include "SubWindow.h"
#include <vector>

class ObjDetector {
public:
	virtual void Detect(int *g_img, std::vector<SubWindow>& objs) = 0;
	virtual ~ObjDetector() { };
};

#endif /* OBJDETECTOR_H_ */
