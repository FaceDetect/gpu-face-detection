/*
 * main.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#include <opencv2/opencv.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#include "DataSet.h"

#include "utils.h"
#include "DecisionStump.h"
#include "GpuDecisionStump.h"
#include "CascadeTrainer.h"
//#include "Feature.h"
//#include <algorithm>
//#include "DecisionStump.h"
//#include "AdaBoost.h"

using namespace cv;
using namespace std;

static void Help() {
	cout << "Usage: ./cascade_trainer <pos_list> <neg_list> <test_pos> <test_neg> <num_weaks*>" << endl;
}

int main( int argc, char **argv ) {

	if (argc < 6) {
		Help();
		return -1;
	}


	DataSet data_set(argv[1], argv[2]);
	DataSet test_set(argv[1], argv[2]);

	vector<int> num_weaks;

	for (int i = 5; i < argc; i++) {
		num_weaks.push_back(atoi(argv[i]));
	}

	CascadeTrainer trainer(data_set, test_set);
	trainer.Train(num_weaks);
}
