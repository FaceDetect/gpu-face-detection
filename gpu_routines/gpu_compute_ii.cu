/*
 * gpu_compute_ii.cu
 *
 *  Created on: May 3, 2013
 *      Author: olehp
 */

#include "gpu_compute_ii.h"
#include "gpu_utils.h"
#include <math.h>
#include <time.h>
#include <iostream>

using namespace std;


__global__ void kernel_ii_rows(int *matr, int *result, int *sq_result, int rows, int cols) {

	int row = threadIdx.x + blockIdx.x * blockDim.x;

	int start_offset = row * cols;
	int val;
	int i;
	int cur_sum = 0, cur_sq_sum = 0;

	if (row < rows) {
		for (i = 0; i < cols; i++) {
			val = matr[start_offset + i];
			cur_sum += val;
			cur_sq_sum += (val * val);

			result[start_offset + i] = cur_sum;
			sq_result[start_offset + i] = cur_sq_sum;
		}
	}
}

__global__ void kernel_ii_cols(int *result, int *sq_result, int rows, int cols) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int i;

	if (col < cols) {
		for (i = 1; i < rows; i++) {
			MATR_VAL(result, i, col, cols) += MATR_VAL(result, i - 1, col, cols);
			MATR_VAL(sq_result, i, col, cols) += MATR_VAL(sq_result, i - 1, col, cols);
		}
	}
}

void gpuComputeII(const int *matr, int *result, int *sq_result, int rows, int cols) {
	int *dev_matr, *dev_result, *dev_sq_result;
	HANDLE_ERROR(cudaMalloc((void **)&dev_matr, rows * cols * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_result, rows * cols * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_sq_result, rows * cols * sizeof(int)));


	HANDLE_ERROR(cudaMemcpy((void *)dev_matr, (void *)matr, rows * cols * sizeof(int), cudaMemcpyHostToDevice));

	dim3 block(512);
	dim3 grid_rows(ceil(rows / 512.0));
	dim3 grid_cols(ceil(cols / 512.0));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	
	kernel_ii_rows<<<grid_rows, block>>>(dev_matr, dev_result, dev_sq_result, rows, cols);
	kernel_ii_cols<<<grid_cols, block>>>(dev_result, dev_sq_result, rows, cols);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
//	cout << "Time elapsed: " << elapsed << endl;

	HANDLE_ERROR(cudaMemcpy((void *)result, (void *)dev_result, cols * rows * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy((void *)sq_result, (void *)dev_sq_result, cols * rows * sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(dev_matr));
	HANDLE_ERROR(cudaFree(dev_result));
}

