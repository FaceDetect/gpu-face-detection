/*
 * main.cpp
 *
 *  Created on: Apr 19, 2013
 *      Author: olehp
 */

#include <iostream>
#include <gtest/gtest.h>
using namespace std;

GTEST_API_ int main(int argc, char **argv) {
	printf("Running main() from gtest_main.cc\n");
	testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}
