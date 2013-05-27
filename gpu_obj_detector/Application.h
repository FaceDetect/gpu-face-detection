/*
 * Application.h
 *
 *  Created on: May 27, 2013
 *      Author: olehp
 */

#ifndef APPLICATION_H_
#define APPLICATION_H_

#include "Media.h"
#include "ObjDetector.h"

#include <string>

class Application {
public:
	Application(Media& source, const char *cascade_path, const char * w_name = "Detection");
	void Run();
	~Application();

private:
	void DetectAndDisplay(cv::Mat& img);

	Media& media;
	ObjDetector *detector;
	const char *window_name;
	HaarCascade haar_cascade;
};

#endif /* APPLICATION_H_ */
