/*
 * prepare_data_set_test.cc
 *
 *  Created on: Apr 21, 2013
 *      Author: olehp
 */

#include <gtest/gtest.h>

#include <vector>
#include <algorithm>

#include "DataSet.h"
#include "constants.h"

using namespace std;


TEST(PrepareDataSet, Size) {
	vector<LabeledImg> images;
	LoadImages("test_pos.txt", "test_neg.txt", images);

	DataSet data_set("test_pos.txt", "test_neg.txt");

	EXPECT_EQ(images.size(), data_set.data.rows);
	EXPECT_EQ(GetFeatureSet().size(), data_set.data.cols);
	EXPECT_EQ(images.size(), data_set.labels.rows);
	EXPECT_EQ(1, data_set.labels.cols);

	int num_pos = count_if(images.begin(), images.end(), [](LabeledImg &limg) { return limg.second == POSITIVE_LABEL; });
	int num_neg = count_if(images.begin(), images.end(), [](LabeledImg &limg) { return limg.second == NEGATIVE_LABEL; });

	EXPECT_EQ(num_pos, count(data_set.labels.begin(), data_set.labels.end(), POSITIVE_LABEL));

	EXPECT_EQ(num_neg, count(data_set.labels.begin(), data_set.labels.end(), NEGATIVE_LABEL));
}

