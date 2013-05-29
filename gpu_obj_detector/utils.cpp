/*
 * utils.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#include "utils.h"
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <rapidxml.h>

using namespace std;
using namespace rapidxml;

void ReadWholeFile(const char *path, char **out_content) {
	FILE *f;
	long file_size;

	if ((f = fopen(path, "rt")) == NULL) {
		printf("NO file\n");
		return;
	}

	fseek(f, 0, SEEK_END);
	file_size = ftell(f);
	fseek(f, 0, SEEK_SET);


	*out_content = new char[file_size];

	fread(*out_content, sizeof(char), file_size, f);
	(*out_content)[file_size - 1] = '\0';
	fclose(f);
}


std::vector<std::string> &Split(const std::string &s, char delim, std::vector<std::string> &elems)
{
	std::stringstream ss(s);
	std::string item;
    while(std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> Split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    return Split(s, delim, elems);
}

void LoadRects(rapidxml::xml_node<> *rect, WeightedRectangle *rects) {

	int i = 0;

	do {
		WeightedRectangle r;
		vector<string> tokens = Split(rect->value(), ' ');

		r.x = atoi(tokens.at(0).c_str());
		r.y = atoi(tokens.at(1).c_str());
		r.w = atoi(tokens.at(2).c_str());
		r.h = atoi(tokens.at(3).c_str());
		r.wg = atoi(tokens.at(4).c_str());

		rects[i++] = r;

	} while((rect = rect->next_sibling()));
}

void LoadFeature(rapidxml::xml_node<>* feature, Feature& f) {

//	f.tilted = atof(feature->first_node("tilted")->value());

	LoadRects(feature->first_node("rects")->first_node(), f.rects);
}


void LoadTrees(rapidxml::xml_node<> *tree, Tree *trees) {
	int i = 0;

	do {
		xml_node<> *root_node = tree->first_node();

		Tree current_tree;

		current_tree.left_val = atof(root_node->first_node("left_val")->value());
		current_tree.right_val = atof(root_node->first_node("right_val")->value());
		current_tree.threshold = atof(root_node->first_node("threshold")->value());
		current_tree.valid = 1;

		LoadFeature(root_node->first_node("feature"), current_tree.feature);

		trees[i++] = current_tree;

	} while((tree = tree->next_sibling()));
}

void LoadStages(rapidxml::xml_node<>* stage, Stage *stages) {

	int i = 0;

	do {
		Stage current_stage;

		current_stage.threshold = atof(stage->first_node("stage_threshold")->value());
		current_stage.parent = atoi(stage->first_node("parent")->value());
		current_stage.next = atoi(stage->first_node("next")->value());
		current_stage.valid = 1;

		LoadTrees(stage->first_node("trees")->first_node(), current_stage.trees);

		stages[i++] = current_stage;

	} while((stage = stage->next_sibling()));

}

void ComputeIIs(const int* input, int* ii, int* ii2, int img_width, int img_height) {
	//	gpuComputeII(grayscaled_bytes, ii, ii2, pic_height, pic_width);
	for (int y = 1; y < img_height + 1; y++) {
		for (int x = 1; x < img_width + 1; x++) {
//			std::cout << x << " " << y << std::endl;
			SetMatrVal(ii, y, x,
					   MatrVal(input, y - 1, x - 1, img_width) -
					   MatrVal(ii, y - 1, x - 1, img_width + 1) +
					   MatrVal(ii, y, x - 1, img_width + 1) +
					   MatrVal(ii, y - 1, x, img_width + 1),
					   img_width + 1);

			SetMatrVal(ii2, y, x,
					   OR_SQR(MatrVal(input, y - 1, x - 1, img_width))  -
					   MatrVal(ii2, y - 1, x - 1, img_width + 1) +
					   MatrVal(ii2, y, x - 1, img_width + 1) +
					   MatrVal(ii2, y - 1, x, img_width + 1),
					   img_width + 1);
		}
	}
}

void LoadCascade(const char *path, HaarCascade& haar_cascade) {

	char *file_content;
	ReadWholeFile(path, &file_content);

	xml_document<> doc;

	doc.parse<0>(file_content);

	xml_node<> *cascade = doc.first_node()->first_node();

	xml_node<> *size = cascade->first_node("size");

	vector<string> tokens = Split(size->value(), ' ');

	haar_cascade.window_width = atoi(tokens.at(0).c_str());
	haar_cascade.window_height = atoi(tokens.at(1).c_str());

	xml_node<> *stages = cascade->first_node("stages");
	LoadStages(stages->first_node(), haar_cascade.stages);

	delete [] file_content;
}

