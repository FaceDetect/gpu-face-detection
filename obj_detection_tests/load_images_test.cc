/*
 * load_images_test.cc
 *
 *  Created on: Apr 21, 2013
 *      Author: olehp
 */

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <exception>

#include "utils.h"
#include "constants.h"

using namespace std;
using namespace cv;



TEST(LoadImagesTest, ImagesNumber) {
	vector<LabeledImg> images;
	LoadImages("test_pos.txt", "test_neg.txt", images);

	ASSERT_FALSE(images.empty());

	ifstream pos("test_pos.txt");
	ifstream neg("test_neg.txt");

	int num_pos = count(istreambuf_iterator<char>(pos), istreambuf_iterator<char>(), '\n');
	int num_neg = count(istreambuf_iterator<char>(neg), istreambuf_iterator<char>(), '\n');

	EXPECT_EQ(num_pos + num_neg, images.size());
	EXPECT_EQ(num_pos, count_if(images.begin(), images.end(), [](LabeledImg &limg) { return limg.second == POSITIVE_LABEL; }));
	EXPECT_EQ(num_neg, count_if(images.begin(), images.end(), [](LabeledImg &limg) { return limg.second == NEGATIVE_LABEL; }));
}

TEST(LoadImagesTest, NoSuchFile) {
	vector<LabeledImg> dummy;

	EXPECT_THROW(LoadImages("not_existing.txt", "test_neg.txt", dummy), std::exception);
	EXPECT_THROW(LoadImages("test_pos.txt", "not_existing.txt", dummy), std::exception);
	EXPECT_THROW(LoadImages("not_existing.txt", "not_existing.txt", dummy), std::exception);
}
