/*
 * main.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include "TrainingData.h"
#include "utils.h"
#include "Feature.h"
#include <algorithm>
#include "DecisionStump.h"
#include "AdaBoost.h"

using namespace cv;
using namespace std;

int main( int argc, char **argv ){

	TrainingData td;
	td.LoadImages("positives.txt", "negatives.txt");
	td.PrepareDataSet();

	AdaBoost ab(td);

	ab.Train(5);

	ENDL
	PrintMatrix(GET_MAT_COL(td.data_set, td.data_set.cols -1));
	ENDL
	PrintMatrix(ab.Classify(td));

    return 0;
}
