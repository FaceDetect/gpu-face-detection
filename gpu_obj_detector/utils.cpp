/*
 * utils.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#include "utils.h"
#include <stdio.h>
#include <sstream>
#include <rapidxml.h>
#include "Feature.h"
#include <cstring>
#include <iostream>

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

void LoadRects(rapidxml::xml_node<> *rect, Rectangle *rects) {

	int i = 0;

	do {
		Rectangle r;
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

	f.tilted = atof(feature->first_node("tilted")->value());

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

void ComputeIIs(const int* input, int* ii, int* ii2, int img_width) {
	//	gpuComputeII(grayscaled_bytes, ii, ii2, pic_height, pic_width);
	for (int y = 1; y < img_width + 1; y++) {
		for (int x = 1; x < img_width + 1; x++) {

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


int shorts_to_int(short s1, short s2) {
	union shorts_int {
		short s[2];
		int i;
	};

	shorts_int si;

	si.s[0] = s1;
	si.s[1] = s2;

	return si.i;

}

void HaarCascadeToArrays(HaarCascade& haar_cascade,
		int4** stages, int4** features, int4** rects, float** weights,
		int *num_stages, int *num_features, int *num_rects) {

	vector<int4> tmp_stages;
	vector<int4> tmp_features;
	vector<int4> tmp_rects;
	vector<float> tmp_weights;

	int features_total = 0;
	int rects_total = 0;

	for (int i = 0; i < HAAR_MAX_STAGES; i++) {
		Stage& stage = haar_cascade.stages[i];
		if (!stage.valid) break;
		int4 res_stage;

		short fid = features_total;

		for (int j = 0; j < HAAR_MAX_TREES; j++) {
			Tree& tree = stage.trees[j];
			if (!tree.valid) break;

			int4 res_feature;
			short rect_id = rects_total;
			for (int k = 0; k < HAAR_MAX_RECTS; k++) {
				Rectangle& rect = tree.feature.rects[k];
				if (rect.wg == 0) break;

				int4 res_rect;
				float res_wg;
				res_rect.w = rect.x;
				res_rect.x = rect.y;
				res_rect.y = rect.w;
				res_rect.z = rect.h;
				res_wg = rect.wg;

				tmp_rects.push_back(res_rect);
				tmp_weights.push_back(res_wg);
				rects_total++;
			}

			res_feature.w = shorts_to_int(rect_id, (short) rects_total - rect_id);
			res_feature.x = reinterpret_cast<int&>(tree.threshold);
			res_feature.y = reinterpret_cast<int&>(tree.left_val);
			res_feature.z = reinterpret_cast<int&>(tree.right_val);
			tmp_features.push_back(res_feature);
			features_total++;
		}

		res_stage.w = shorts_to_int(fid, (short) features_total - fid);
		res_stage.x = reinterpret_cast<int&>(stage.threshold);
		res_stage.y = stage.parent;
		res_stage.z = stage.next;

		tmp_stages.push_back(res_stage);
	}

//	cout << "Total stages size: " << tmp_stages.size() * sizeof(int4) << endl;
//	cout << "Total features size: " << tmp_features.size() * sizeof(int4) << endl;
//	cout << "Total rects size: " << tmp_rects.size() * sizeof(int4) << endl;
//	cout << "Total weights size: " << tmp_weights.size() * sizeof(float) << endl << endl;


	 if (num_stages != NULL) (*num_stages) = tmp_stages.size();
	 if (num_features != NULL) (*num_features) = tmp_features.size();
	 if (num_rects != NULL) (*num_rects) = tmp_rects.size();


	(*stages) = new int4[tmp_stages.size()];
	(*features) = new int4[tmp_features.size()];
	(*rects) = new int4[tmp_rects.size()];
	(*weights) = new float[tmp_weights.size()];


	memcpy((*stages), &tmp_stages[0], sizeof(int4) * tmp_stages.size());
	memcpy((*features), &tmp_features[0], sizeof(int4) * tmp_features.size());
	memcpy((*rects), &tmp_rects[0], sizeof(int4) * tmp_rects.size());
	memcpy((*weights), &tmp_weights[0], sizeof(float) * tmp_weights.size());

}

