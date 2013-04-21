/*
 * sqr_test.cc
 *
 *  Created on: Apr 19, 2013
 *      Author: olehp
 */

#include <gtest/gtest.h>

#include "utils.h"

TEST(Sqr, PositiveNumbers) {
	EXPECT_EQ(1, SQR(1));
	EXPECT_EQ(4, SQR(2));
	EXPECT_EQ(9, SQR(3));
	EXPECT_EQ(169, SQR(13));
}

TEST(Sqr, NegativeNumbers) {
	EXPECT_EQ(1, SQR(1));
	EXPECT_EQ(4, SQR(2));
	EXPECT_EQ(9, SQR(3));
	EXPECT_EQ(169, SQR(13));
}

TEST(Sqr, Zero) {
	EXPECT_EQ(0, SQR(0));
}
