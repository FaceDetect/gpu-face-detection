/*
 * GpuDecisionStump.h
 *
 *  Created on: May 30, 2013
 *      Author: olehp
 */

#ifndef GPUDECISIONSTUMP_H_
#define GPUDECISIONSTUMP_H_

#include "DataSet.h"

class GpuDecisionStump {
public:

#ifdef __CUDACC__
	__host__ __device__ GpuDecisionStump() { }
#else
	GpuDecisionStump() { }
#endif

	GpuDecisionStump(int i, float threshold, bool gt);

	void Train(const DataSet& data_set, const cv::Mat_<float>& W);
	void Classify(const Data& data, cv::Mat_<label_t>& labels);


	float threshold;
	int i_feature;
	bool gt;
	float wg_err;
};

#endif /* GPUDECISIONSTUMP_H_ */
