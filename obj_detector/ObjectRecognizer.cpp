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
#include <string.h>
#include <math.h>
#include <time.h>

using namespace rapidxml;
using namespace std;
using namespace cv;



ObjectRecognizer::ObjectRecognizer() :
		pic_width(-1),
		pic_height(-1),
		grayscaled_pic(0, 0),
		grayscaled_bytes(NULL),
		ii(NULL),
		ii2(NULL) {
}

ObjectRecognizer::~ObjectRecognizer() {
}

void ObjectRecognizer::LoadHaarCascade(const char *path) {

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
	LoadStages(stages->first_node());

	delete [] file_content;
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

void ObjectRecognizer::Recognize() {
//	const clock_t begin_time = clock();

	ComputeIntegralImages();

//	std::cout << endl << "Time elapsed: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

	double scale = 1.0;

	int width = haar_cascade.window_width;
	int height = haar_cascade.window_height;

	while (OR_MIN(width, height) <= OR_MIN(pic_width, pic_height)) {

		int x_step = OR_MAX(1, OR_MIN(4, floor(width / 10)));
		int y_step = OR_MAX(1, OR_MIN(4, floor(height / 10)));

		double inv = 1.0 / (width * height);

		for (int y = 0; y < pic_height - height; y += y_step) {
			for (int x = 0; x < pic_width - width; x += x_step) {

				double mean = RectSum(ii, x, y, width, height) * inv;
				double variance = RectSum(ii2, x, y, width, height) * inv - OR_SQR(mean);

				double std_dev = 1;

				if (variance >= 0)
					std_dev = sqrt(variance);


				if (std_dev < 10)
					continue;


				if (StagesPass(x, y, scale, inv, std_dev)) {
					cout << x << " " << y << " " << width << " " << height << " " << scale << " ";
				}
			}
		}

		scale = scale * 1.2;
		width = (int)(haar_cascade.window_width * scale);
		height = (int)(haar_cascade.window_height * scale);
	}
}

void ObjectRecognizer::LoadImage(const char* path) {
	//FIBITMAP *colourful_pic = FreeImage_Load(FreeImage_GetFIFFromFilename(path), path, JPEG_ACCURATE);
	//grayscaled_pic = FreeImage_ConvertToGreyscale(colourful_pic);
	//FreeImage_Unload(colourful_pic);

	//FreeImage_FlipVertical(grayscaled_pic);

	grayscaled_pic = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

	pic_width = grayscaled_pic.cols;
	pic_height = grayscaled_pic.rows;
	grayscaled_bytes = grayscaled_pic.ptr<int>();

//	cout << endl;
//	cout << endl;
//
//	for (int i = 0; i < pic_height; i++) {
//		for (int j = 0; j < pic_width; j++)
//			cout << (int)grayscaled_bytes[i * pic_width + j] << " ";
//
//		cout << endl;
//	}
//
//	cout << endl;
//	cout << endl;

	ii = new int[pic_width * pic_height];
	ii2 = new int[pic_width * pic_height];
}

void ObjectRecognizer::UnloadImage() {
	delete [] ii;
	delete [] ii2;
}

void ObjectRecognizer::ComputeIntegralImages() {

	for (int y = 0; y < pic_height; y++) {
		for (int x = 0; x < pic_width; x++) {
			SetMatrVal(ii, y, x,
					   MatrVal(grayscaled_bytes, y, x) -
					   MatrVal(ii, y - 1, x - 1) +
					   MatrVal(ii, y, x - 1) +
					   MatrVal(ii, y - 1, x));

			SetMatrVal(ii2, y, x,
					   MatrVal(grayscaled_bytes, y, x) * MatrVal(grayscaled_bytes, y, x) -
					   MatrVal(ii2, y - 1, x - 1) +
					   MatrVal(ii2, y, x - 1) +
					   MatrVal(ii2, y - 1, x));
		}
	}
}

bool ObjectRecognizer::StagesPass(int x, int y, double scale, double inv, double std_dev) {

	for (uint i = 0; i < haar_cascade.stages.size(); i++) {
		Stage &stage = haar_cascade.stages[i];
		double tree_sum = TreesPass(stage, x, y, scale, inv, std_dev);
		if (tree_sum < stage.threshold) {
			return false;
		}
	}

	return true;
}

double ObjectRecognizer::TreesPass(Stage &stage, int x, int y, double scale, double inv, double std_dev) {

    double tree_sum = 0;

    for (uint i = 0; i < stage.trees.size(); i++) {
    	Tree &tree = stage.trees[i];
        double rects_sum = RectsPass(tree, x, y, scale) * inv;

        if (rects_sum < tree.threshold * std_dev)
            tree_sum = tree_sum + tree.left_val;
        else
            tree_sum = tree_sum + tree.right_val;
    }

    return tree_sum;
}

double ObjectRecognizer::RectsPass(Tree &tree, int x, int y, double scale) {
	double rects_sum = 0;
	for (uint i = 0; i < tree.feature.rects.size(); i++) {
		Rectangle &rect = tree.feature.rects[i];
		rects_sum = rects_sum + RectSum(ii, x + (int)(rect.x * scale),
											y + (int)(rect.y * scale),
											(int)(rect.w * scale),
											(int)(rect.h * scale)) * rect.wg;
	}

	return rects_sum;
}


inline int ObjectRecognizer::RectSum(int* ii, int x, int y, int w, int h) {

	return MatrVal(ii, y - 1, x - 1) +
		   MatrVal(ii, y + h - 1, x + w - 1) -
		   MatrVal(ii, y - 1, x + w - 1) -
		   MatrVal(ii, y + h - 1, x - 1);

//	return MatrVal(ii, y, x) +
//		   MatrVal(ii, y + w, x + h) -
//		   MatrVal(ii, y, x + w) -
//		   MatrVal(ii, y + h, x);
}
