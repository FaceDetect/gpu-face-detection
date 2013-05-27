/*
 * ImgMedia.h
 *
 *  Created on: May 27, 2013
 *      Author: olehp
 */

#ifndef IMGMEDIA_H_
#define IMGMEDIA_H_

#include "Media.h"

class ImgMedia : public Media{
public:
	ImgMedia(const char *path);
	virtual void GetFrame(cv::Mat& frame);
	virtual ~ImgMedia() { }
private:
	cv::Mat img;
};

#endif /* IMGMEDIA_H_ */
