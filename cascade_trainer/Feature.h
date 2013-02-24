/*
 * Feature.h
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#ifndef FEATURE_H_
#define FEATURE_H_

#include "utils.h"

class Feature
{
public:
	Feature();
	Feature( int offset, bool _tilted,
		int x0, int y0, int w0, int h0, float wt0,
		int x1, int y1, int w1, int h1, float wt1,
		int x2 = 0, int y2 = 0, int w2 = 0, int h2 = 0, float wt2 = 0.0F );
//	float calc( const Mat &sum, const Mat &tilted, size_t y) const;
//	void write( FileStorage &fs ) const;
//

	struct
	{
		Rect r;
		float weight;
	} rect[HAAR_MAX_FEATURES];

	struct
	{
		int p0, p1, p2, p3;
	} fastRect[HAAR_MAX_FEATURES];
};



#endif /* FEATURE_H_ */
