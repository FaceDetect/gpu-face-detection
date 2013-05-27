/*
 * WebCamMedia.h
 *
 *  Created on: May 27, 2013
 *      Author: olehp
 */

#ifndef WEBCAMMEDIA_H_
#define WEBCAMMEDIA_H_

#include "Media.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class WebCamMedia : public Media{
public:
	WebCamMedia(int w, int h, int web_cam_id = -1);
	virtual void GetFrame(cv::Mat& frame);
	virtual ~WebCamMedia();
private:
	CvCapture *capture;
};

#endif /* WEBCAMMEDIA_H_ */
