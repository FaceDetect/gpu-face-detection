/*
 * gpu_prepare_data_set.h
 *
 *  Created on: May 15, 2013
 *      Author: olehp
 */

#ifndef GPUPREPAREDATASET_H_
#define GPUPREPAREDATASET_H_

#include "DataSet.h"

void gpu_prepare_data_set(std::vector<LabeledImg>& imgs,
						  const std::vector<Feature>& feature_set,
						  cv::Mat_<label_t>& labels,
						  Data& data);

#endif /* GPUPREPAREDATASET_H_ */
