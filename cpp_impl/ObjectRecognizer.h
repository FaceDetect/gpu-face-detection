/*
 * ObjectRecognizer.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef OBJECTRECOGNIZER_H_
#define OBJECTRECOGNIZER_H_

#include "HaarCascade.h"
#include <rapidxml-1.13/rapidxml.hpp>
#include <FreeImage.h>

class ObjectRecognizer {

public:
	ObjectRecognizer();
	~ObjectRecognizer();
	void LoadHaarCascade(const char *path);
	void LoadImage(const char *path);
	void Recognize();
private:

	void LoadStages(rapidxml::xml_node<> *stage);
	void LoadTrees(rapidxml::xml_node<> *tree, std::vector<Tree> &trees);
	void LoadFeature(rapidxml::xml_node<> *feature, Feature &f);
	void LoadRects(rapidxml::xml_node<> *rect, std::vector<Rectangle> &rects);


	int pic_width;
	int pic_height;
	FIBITMAP *colourful_pic;
	FIBITMAP *grayscaled_pic;

	HaarCascade haar_cascade;
};

#endif /* OBJECTRECOGNIZER_H_ */
