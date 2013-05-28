/*
 * GpuObjDetector.h
 *
 *  Created on: May 25, 2013
 *      Author: olehp
 */

#ifndef GPUOBJDETECTOR_H_
#define GPUOBJDETECTOR_H_

#include "ObjDetector.h"
#include "SubWindow.h"

#include <vector>

class GpuObjDetector : public ObjDetector {
public:
	GpuObjDetector(int w, int h, HaarCascade& cascade);
	virtual void Detect(int *g_img, std::vector<SubWindow>& objs);
	virtual ~GpuObjDetector();
private:

	void DetectAtSubwindows(std::vector<SubWindow>& subwindows);
	void GpuComputeII();
	int *dev_img;
	int *dev_ii;
	int *dev_ii2;
	SubWindow *dev_subwindows;
	std::vector<SubWindow> all_subwindows;
	int img_width;
	int img_height;
	HaarCascade haar_cascade;

	int img_mem_size;
	int ii_mem_size;
};

#endif /* GPUOBJDETECTOR_H_ */
