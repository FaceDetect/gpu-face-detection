/*
 * main.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#include "Application.h"
#include "ImgMedia.h"
#include "WebCamMedia.h"

#include <iostream>
using namespace std;


static void Help() {
	cout << "Usage: ./gpu_obj_detector \n\t--data <data_file>\n\t(--image <path>|--camera <width> <height> <camera_id>)" << endl;
}

int main(int argc, char **argv)
{
	if (argc == 1) {
		Help();
		return -1;
	}

	Media *media;
	string cascade_path;

	for (int i = 1; i < argc; i++) {
		if (string(argv[i]) == "--data") {
			cascade_path = string(argv[++i]);
		} else if (string(argv[i]) == "--camera") {
			int camera_width = atoi(argv[++i]);
			int camera_height = atoi(argv[++i]);
			int camera_id = atoi(argv[++i]);

			media = new WebCamMedia(camera_width, camera_height, camera_id);
		} else if (string(argv[i]) == "--image") {
			media = new ImgMedia(argv[++i]);
		}
	}

	Application app(*media, cascade_path.c_str());
	app.Run();


	delete media;
	return 0;

}
