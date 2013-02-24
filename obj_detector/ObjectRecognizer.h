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
	void UnloadImage();
	void Recognize();
private:

	void ComputeIntegralImages();
	void LoadStages(rapidxml::xml_node<> *stage);
	void LoadTrees(rapidxml::xml_node<> *tree, std::vector<Tree> &trees);
	void LoadFeature(rapidxml::xml_node<> *feature, Feature &f);
	void LoadRects(rapidxml::xml_node<> *rect, std::vector<Rectangle> &rects);

	bool StagesPass(int x, int y, double scale, double inv, double stdDev);
	double TreesPass(Stage &stage, int x, int y, double scale, double inv, double stdDev);
	double RectsPass(Tree &tree, int x, int y, double scale);

	inline int RectSum(int *ii, int x, int y, int w, int h);
	template<typename T>
	inline T MatrVal(T *arr, int row, int col) {

		return ((row == -1) || (col == -1)) ? 0 : arr[row * pic_width + col];
	}

	template<typename T>
	inline void SetMatrVal(T *arr, int row, int col, T val) {
		arr[row * pic_width + col] = val;
	}


	int pic_width;
	int pic_height;
	FIBITMAP *grayscaled_pic;
	BYTE *grayscaled_bytes;
	int *ii;
	int *ii2;

	HaarCascade haar_cascade;
};

#endif /* OBJECTRECOGNIZER_H_ */
