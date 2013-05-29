/*
 * ObjectRecognizer.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef OBJECTRECOGNIZER_H_
#define OBJECTRECOGNIZER_H_

#include "ObjDetector.h"

class CpuObjDetector : public ObjDetector {

public:
	CpuObjDetector(int w, int h, HaarCascade& cascade);
	virtual ~CpuObjDetector();
	virtual void Detect(const int *g_img, std::vector<Rectangle>&  objs);
private:

	void ComputeIntegralImages();
	bool StagesPass(int x, int y, double scale, double inv, double stdDev);
	inline int RectSum(int *ii, int x, int y, int w, int h);

	int pic_width;
	int pic_height;
	int *grayscaled_bytes;
	int *ii;
	int *ii2;

	HaarCascade haar_cascade;
};

#endif /* OBJECTRECOGNIZER_H_ */
