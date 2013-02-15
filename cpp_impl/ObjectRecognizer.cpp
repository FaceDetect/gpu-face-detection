/*
 * ObjectRecognizer.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#include "ObjectRecognizer.h"
#include "utils.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

using namespace rapidxml;
using namespace std;



ObjectRecognizer::ObjectRecognizer() {
	FreeImage_Initialise(true);
}

ObjectRecognizer::~ObjectRecognizer() {
	FreeImage_DeInitialise();
}

void ObjectRecognizer::LoadHaarCascade(const char *path) {

	char *file_content;
	ReadWholeFile(path, &file_content);

	xml_document<> doc;

	doc.parse<0>(file_content);
	cout << "FIRST NODE: " << doc.first_node()->name() << endl;

	xml_node<> *cascade = doc.first_node()->first_node();

	xml_node<> *size = cascade->first_node("size");

	vector<string> tokens = Split(size->value(), ' ');

	haar_cascade.window_width = atoi(tokens.at(0).c_str());
	haar_cascade.window_height = atoi(tokens.at(1).c_str());

	xml_node<> *stages = cascade->first_node("stages");
	LoadStages(stages->first_node());


}

void ObjectRecognizer::LoadStages(rapidxml::xml_node<>* stage) {

	do {
		Stage current_stage;

		current_stage.threshold = atof(stage->first_node("stage_threshold")->value());
		current_stage.parent = atoi(stage->first_node("parent")->value());
		current_stage.next = atoi(stage->first_node("next")->value());

		LoadTrees(stage->first_node("trees")->first_node(), current_stage.trees);

		haar_cascade.stages.push_back(current_stage);

	} while((stage = stage->next_sibling()));

}

void ObjectRecognizer::LoadTrees(rapidxml::xml_node<> *tree, vector<Tree> &trees) {

	do {
		xml_node<> *root_node = tree->first_node();

		Tree current_tree;

		current_tree.left_val = atof(root_node->first_node("left_val")->value());
		current_tree.right_val = atof(root_node->first_node("right_val")->value());
		current_tree.threshold = atof(root_node->first_node("threshold")->value());

		LoadFeature(root_node->first_node("feature"), current_tree.feature);

		trees.push_back(current_tree);

	} while((tree = tree->next_sibling()));
}

void ObjectRecognizer::LoadFeature(rapidxml::xml_node<>* feature, Feature& f) {

	f.tilted = atof(feature->first_node("tilted")->value());

	LoadRects(feature->first_node("rects")->first_node(), f.rects);
}

void ObjectRecognizer::Recognize() {
}

void ObjectRecognizer::LoadRects(rapidxml::xml_node<> *rect, std::vector<Rectangle> &rects) {

	do {
		Rectangle r;
		vector<string> tokens = Split(rect->value(), ' ');

		r.x = atoi(tokens.at(0).c_str());
		r.y = atoi(tokens.at(1).c_str());
		r.w = atoi(tokens.at(2).c_str());
		r.h = atoi(tokens.at(3).c_str());
		r.wg = atoi(tokens.at(4).c_str());

		rects.push_back(r);

	} while((rect = rect->next_sibling()));
}


void ObjectRecognizer::LoadImage(const char* path) {
	colourful_pic = FreeImage_Load(FreeImage_GetFIFFromFilename(path), path, 0);
	grayscaled_pic = FreeImage_ConvertToGreyscale(colourful_pic);
	pic_width = FreeImage_GetPitch(grayscaled_pic);
	pic_height = FreeImage_GetHeight(grayscaled_pic);

	FreeImage_Save(FreeImage_GetFIFFromFilename("test.jpg"), grayscaled_pic, "test.jpg", 0);



}
