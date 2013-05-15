/*
 * main.cpp
 *
 *  Created on: Feb 23, 2013
 *      Author: olehp
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include "DataSet.h"
#include "utils.h"
//#include "Feature.h"
//#include <algorithm>
//#include "DecisionStump.h"
//#include "AdaBoost.h"

using namespace cv;
using namespace std;

int main( int argc, char **argv ) {

	const clock_t begin_time = clock();

	DataSet("one_pos.txt", "one_pos.txt");

	std::cout << endl << "Time elapsed: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

    return 0;
}
