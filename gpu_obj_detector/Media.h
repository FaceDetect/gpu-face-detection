/*
 * Media.h
 *
 *  Created on: May 27, 2013
 *      Author: olehp
 */

#ifndef MEDIA_H_
#define MEDIA_H_

#include <opencv2/opencv.hpp>

class Media {
public:
	virtual void GetFrame(cv::Mat& frame) = 0;
	inline int GetWidth() { return width; }
	inline int GetHeight() { return height; }

	virtual ~Media() { }
protected:
	int width;
	int height;
};

#endif /* MEDIA_H_ */
