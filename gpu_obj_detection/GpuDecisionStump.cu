/*
 * GpuDecisionStump.cpp
 *
 *  Created on: May 30, 2013
 *      Author: olehp
 */

#include "GpuDecisionStump.h"

#include "gpu_utils.h"


//using namespace thrust;

#include <iostream>

using namespace std;


#define MAX_THREAD_TRAIN 64

GpuDecisionStump::GpuDecisionStump(int i, float threshold, bool gt) :
	i_feature(i),
	threshold(threshold),
	gt(gt) {

}

void GpuDecisionStump::Classify(const Data& data, cv::Mat_<label_t>& labels) {
	labels.create(data.rows, 1);

	for (int i = 0; i < data.rows; i++) {
		labels(i, 0) = (label_t) ((gt) ? (data(i, i_feature) > threshold) : (data(i, i_feature) <= threshold));
	}
}

__global__ void kernel_gen_stumps(const float *data_col, GpuDecisionStump *stumps, int i_feature, int col_size) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < col_size) {
		stumps[i].i_feature = i_feature;
		stumps[i].gt = true;
		stumps[i].threshold = data_col[i];

		stumps[col_size + i].i_feature = i_feature;
		stumps[col_size + i].gt = false;
		stumps[col_size + i].threshold = data_col[i];

	}
}

__global__ void kernel_calc_wg_err(const float *data_col, GpuDecisionStump *col_stumps, const label_t *labels, const float *W, int col_size) {

	__shared__ float wg_err_buf[MAX_THREAD_TRAIN];


    int tid = threadIdx.x;
    int bid = blockIdx.x;

    wg_err_buf[tid] = 0;


    int i = tid;

    if (col_stumps[bid].gt) {
    	while (i < col_size) {
    		wg_err_buf[tid] += (labels[i] != (label_t)(data_col[i] > col_stumps[bid].threshold)) * W[i];
    		i += MAX_THREAD_TRAIN;
    	}
    } else {
    	while (i < col_size) {
    		wg_err_buf[tid] += (labels[i] != (label_t)(data_col[i] <= col_stumps[bid].threshold)) * W[i];
    		i += MAX_THREAD_TRAIN;
    	}
    }

     __syncthreads();

    for (int s = MAX_THREAD_TRAIN / 2; s > 0; s /= 2) {
    	if (tid < s) {
    		wg_err_buf[tid] += wg_err_buf[tid + s];
    	}

    	__syncthreads();
    }

    if (tid == 0) {
    	col_stumps[bid].wg_err = wg_err_buf[0];
    }
}

__global__ void kernel_get_best_stump(GpuDecisionStump *stumps, int num_stumps) {

	__shared__ GpuDecisionStump stumps_buf[MAX_THREAD_TRAIN];

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	stumps_buf[tid].wg_err = FLT_MAX;

	if (i < num_stumps) {
		stumps_buf[tid] = stumps[i];
	}

	__syncthreads();

	for (int s = MAX_THREAD_TRAIN / 2; s > 0; s /= 2) {
		if (tid < s) {
			stumps_buf[tid] = (stumps_buf[tid].wg_err > stumps_buf[tid + s].wg_err)
							  ? stumps_buf[tid + s]
							  : stumps_buf[tid];
		}

		__syncthreads();
	}

	if (tid == 0) {
		stumps[bid] = stumps_buf[0];
	}
}

void GetBestStump(GpuDecisionStump *dev_stumps, int num_stumps, GpuDecisionStump *result, bool is_result_dev) {

	int num_blocks = ceilf((float) num_stumps / MAX_THREAD_TRAIN);

	kernel_get_best_stump<<<num_blocks, MAX_THREAD_TRAIN>>>(dev_stumps, num_stumps);

	if (num_blocks != 1) {
		GetBestStump(dev_stumps, num_blocks, result, is_result_dev);
	} else {
		if (is_result_dev)
			HANDLE_ERROR(cudaMemcpy(result, dev_stumps, sizeof(GpuDecisionStump), cudaMemcpyDeviceToDevice));
		else
			HANDLE_ERROR(cudaMemcpy(result, dev_stumps, sizeof(GpuDecisionStump), cudaMemcpyDeviceToHost));
	}
}

void GpuDecisionStump::Train(const DataSet& data_set, const cv::Mat_<float>& W) {

	int row_size = data_set.data.cols;
	int col_size = data_set.data.rows;


	GpuDecisionStump *dev_row_stumps;
	GpuDecisionStump *dev_col_stumps;
	float *dev_data_col;
	float *dev_W;
	label_t *dev_labels;

	HANDLE_ERROR(cudaMalloc(&dev_row_stumps, sizeof(GpuDecisionStump) * row_size));
	HANDLE_ERROR(cudaMalloc(&dev_col_stumps, sizeof(GpuDecisionStump) * col_size * 2));
	HANDLE_ERROR(cudaMalloc(&dev_data_col, sizeof(float) * col_size));
	HANDLE_ERROR(cudaMalloc(&dev_W, sizeof(float) * W.rows));
	HANDLE_ERROR(cudaMalloc(&dev_labels, sizeof(float) * data_set.labels.rows));

	HANDLE_ERROR(cudaMemcpy(dev_W, W.ptr<float>(), sizeof(float) * W.rows, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_labels, data_set.labels.ptr<label_t>(), sizeof(label_t) * data_set.labels.rows, cudaMemcpyHostToDevice));


	int num_blocks = ceilf((float) col_size / MAX_THREAD_TRAIN);

	for (int i = 0; i < data_set.data.cols; i++) {

		cv::Mat_<float> col = data_set.data.col(i).clone();

		cout << i << "/" << row_size << " column processed" << "\r";

		HANDLE_ERROR(cudaMemcpy(dev_data_col, col.ptr<float>(), sizeof(float) * col_size, cudaMemcpyHostToDevice));

		kernel_gen_stumps<<<num_blocks, MAX_THREAD_TRAIN>>>(dev_data_col, dev_col_stumps, i, col_size);
		kernel_calc_wg_err<<<col_size * 2, MAX_THREAD_TRAIN>>>(dev_data_col, dev_col_stumps, dev_labels, dev_W, col_size);


		GetBestStump(dev_col_stumps, col_size * 2, dev_row_stumps + i, true);

	}

	GetBestStump(dev_row_stumps, row_size, this, false);

	HANDLE_ERROR(cudaFree(dev_row_stumps));
	HANDLE_ERROR(cudaFree(dev_col_stumps));
	HANDLE_ERROR(cudaFree(dev_data_col));
	HANDLE_ERROR(cudaFree(dev_W));
	HANDLE_ERROR(cudaFree(dev_labels));
}


