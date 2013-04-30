/*
 * utils.h
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <termios.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>

#include "Feature.h"

typedef int label_t;
typedef std::pair<cv::Mat_<float>, label_t> LabeledImg;

#define GET_MAT_COL(mat, i_col) \
	(mat).colRange((i_col), (i_col) + 1)

#define ENDL std::cout << std::endl;
#define SINGLE_SUM 0
#define SQUARED_SUM 1

int mygetch();
void GenerateFeatures(std::vector<Feature>& features);
void LoadImages(const char * pos_list, const char * neg_list, std::vector<LabeledImg> &container);
std::vector<Feature>& GetFeatureSet();


template<typename T>
inline T sqr(const T &arg) {
	return arg * arg;
}

template <typename T>
void ToIntegralImage(cv::Mat_<T> &mat, int mode) {
	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {

			T p4 = mat(y, x);
			T p3 = (x == 0) ? 0 : mat(y, x - 1);
			T p2 = (y == 0) ? 0 : mat(y - 1, x);
			T p1 = ((x == 0) || (y == 0)) ? 0 : mat(y - 1, x - 1);

			mat(y, x) = ((mode == SQUARED_SUM) ? sqr(p4) : p4) - p1 + p3 + p2;
		}
	}
}

template<class T>
inline T Mean(const cv::Mat_<T> &mat) {
	return sum(mat).val[0] / (mat.rows * mat.cols);
}

template<class T>
float StdDev(const cv::Mat_<T> &mat) {
	T sqr_sum = 0;

	for(const T &i : mat)
		sqr_sum += sqr(i);

	float variance = sqr_sum / (mat.rows * mat.cols) - sqr(Mean(mat));

	return (variance < 0) ? sqrt(-variance) : sqrt(variance);
}

template<typename T>
void NormalizeMat(cv::Mat_<T> &mat) {

	float std_dev = StdDev(mat);


	if (std_dev != 0)
		for (T &i : mat) i = i / std_dev;
}

template <class T>
bool MatrEq(const cv::Mat_<T> &mat1, const cv::Mat_<T> &mat2) {
	if ((mat1.rows != mat2.rows) || (mat1.cols != mat2.cols)) return false;

	cv::MatConstIterator_<T> it1 = mat1.begin();
	cv::MatConstIterator_<T> it2 = mat2.begin();

	for (; it1 != mat1.end(); it1++, it2++)
		if (*it1 != *it2) return false;

	return true;
}


template <class T>
void PrintMatrix(const cv::Mat_<T> &mat) {
	for (int row = 0; row < mat.rows; row++) {
		for (int col = 0; col < mat.cols; col++) {
			std::cout << mat(row, col) << "\t";
		}
		std::cout << std::endl;
	}
}


template <class InputIterator, class OutputIterator, class Transformator>
OutputIterator relaxed_transform(InputIterator first, InputIterator last,
                                 OutputIterator result, Transformator trans) {
	for (; first != last; ++first, ++result)
		*result = trans(*first);

	return result;
}

#endif /* UTILS_H_ */
