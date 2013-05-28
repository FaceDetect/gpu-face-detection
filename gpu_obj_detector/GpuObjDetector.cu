/*
 * GpuObjDetector.cpp
 *
 *  Created on: May 25, 2013
 *      Author: olehp
 */

#include "GpuObjDetector.h"
#include "gpu_utils.h"
#include "utils.h"

#include <iostream>
#include <algorithm>

#define MAX_THREAD 416

using namespace std;


__constant__ __align__(4) char stage_buf[sizeof(Stage)];

__device__ inline int RectSum(const int* ii, int x, int y, int w, int h, int ii_width) {

	return ii[y * ii_width + x] +
		   ii[(y + h) * ii_width + x + w] -
		   ii[y * ii_width + x + w] -
		   ii[(y + h) * ii_width + x];
}



__global__ void kernel_ii_rows(const int *matr, int *result, int *sq_result, int rows, int cols) {

	int row = threadIdx.x + blockIdx.x * blockDim.x;

	int img_start_offset = row * cols;
	int ii_start_offset = (row + 1) * (cols + 1) + 1;

	int val;
	int i;
	int cur_sum = 0, cur_sq_sum = 0;

	if (row < rows) {

		for (i = 0; i < cols; i++) {
			val = matr[img_start_offset + i];
			cur_sum += val;
			cur_sq_sum += (val * val);

			result[ii_start_offset + i] = cur_sum;
			sq_result[ii_start_offset + i] = cur_sq_sum;
		}
	}
}

__global__ void kernel_ii_cols(int *result, int *sq_result, int rows, int cols) {

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int i;
	int ii_start_offset = (cols + 1) + col + 1;
	if (col < cols) {
		for (i = 1; i < rows; i++) {
			result[ii_start_offset + i * (cols + 1)] += result[ii_start_offset + (i - 1) * (cols + 1)];
			sq_result[ii_start_offset + i * (cols + 1)] += sq_result[ii_start_offset + (i - 1) * (cols + 1)];

		}
	}
}


void GpuObjDetector::GpuComputeII() {

	HANDLE_ERROR(cudaMemset(dev_ii, 0, ii_mem_size));
	HANDLE_ERROR(cudaMemset(dev_ii2, 0, ii_mem_size));

	dim3 block(512);
	dim3 grid_rows(ceil(img_height / 512.0));
	dim3 grid_cols(ceil(img_width / 512.0));

	kernel_ii_rows<<<grid_rows, block>>>(dev_img, dev_ii, dev_ii2, img_height, img_width);
	kernel_ii_cols<<<grid_cols, block>>>(dev_ii, dev_ii2, img_height, img_width);
}


__global__ void kernel_detect_objs(int num_stage,
								   const int *ii,
								   const int *ii2,
								   int ii_width,
								   int ii_height,
								   SubWindow *subwindows,
								   int *is_valid,
								   int num_subwindows) {
	// 244 216 123 123 6.19174

	int i_subwindow = threadIdx.x + blockIdx.x * blockDim.x;

	if (!(i_subwindow < num_subwindows)) return;

	float scale = subwindows[i_subwindow].scale;
	int x = subwindows[i_subwindow].x;
	int y = subwindows[i_subwindow].y;
	int width = subwindows[i_subwindow].w;
	int height = subwindows[i_subwindow].h;

	float inv = 1.f / (height * width);
	float mean = RectSum(ii, x, y, width, height, ii_width) * inv;
	float variance = abs(RectSum(ii2, x, y, width, height, ii_width) * inv - mean * mean);

	float std_dev = sqrtf(variance);

	Stage &stage = (Stage&)stage_buf;

	float tree_sum = 0;

	for (int j = 0; j < HAAR_MAX_TREES; j++) {
		Tree& tree = stage.trees[j];
		if (!tree.valid) break;

		float rects_sum = 0;

		for (int k = 0; k < HAAR_MAX_RECTS; k++) {
			Rectangle &rect = tree.feature.rects[k];
			if (rect.wg == 0) break;

			rects_sum = rects_sum + RectSum(ii, x + (int)(rect.x * scale),
												y + (int)(rect.y * scale),
												(int)(rect.w * scale),
												(int)(rect.h * scale), ii_width) * rect.wg;
		}

		tree_sum += ((rects_sum * inv < tree.threshold * std_dev) ? tree.left_val : tree.right_val);
	}


	is_valid[i_subwindow] = (tree_sum >= stage.threshold);

//	if (tree_sum < stage.threshold) {
//		subwindows[i_subwindow].is_object = 0;
//	}

}

bool isNonObject(const SubWindow& s) {
	return !s.is_object;
}

__global__ void kernel_compact_arrays(const SubWindow *subwindows_in,
									  SubWindow *subwindows_out,
									  int *is_valid,
									  int *indexes,
									  int num) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if ((i < num) && is_valid[i]) {

		subwindows_out[indexes[i] - 1] = subwindows_in[i];

	}
}

inline int GetNumBlocks(int n) {
	return ceilf((float) n / MAX_THREAD);
}

void GpuObjDetector::CompactArrays(int& num_subwindows) {
	CUDPPHandle scan_plan;
	cudppPlan(lib, &scan_plan, scan_conf, num_subwindows, 1, 0);
	cudppScan(scan_plan, dev_indexes, dev_is_valid, num_subwindows);

	kernel_compact_arrays<<<GetNumBlocks(num_subwindows), MAX_THREAD>>>(dev_subwindows_in,
																		dev_subwindows_out,
																		dev_is_valid,
																		dev_indexes,
																		num_subwindows);

	cudaMemcpy(&num_subwindows, (dev_indexes + num_subwindows - 1), sizeof(int), cudaMemcpyDeviceToHost);

	SubWindow *tmp;
	tmp = dev_subwindows_in;
	dev_subwindows_in = dev_subwindows_out;
	dev_subwindows_out = tmp;
}

void GpuObjDetector::DetectAtSubwindows(vector<SubWindow>& subwindows) {


	int num_subwindows = subwindows.size();
	HANDLE_ERROR(cudaMemcpy(dev_subwindows_in, &subwindows[0], sizeof(SubWindow) * num_subwindows, cudaMemcpyHostToDevice));

	for (int i = 0; i < HAAR_MAX_STAGES; i++) {

//		int num_subwindows = subwindows.size();
		int num_blocks = ceilf((float) num_subwindows / MAX_THREAD);

//		HANDLE_ERROR(cudaMemcpy(dev_subwindows_in, &subwindows[0], sizeof(SubWindow) * num_subwindows, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpyToSymbol(stage_buf, &haar_cascade.stages[i], sizeof(Stage)));

		kernel_detect_objs<<<num_blocks, MAX_THREAD>>>(i,
											dev_ii,
											dev_ii2,
											img_width + 1,
											img_height + 1,
											dev_subwindows_in,
											dev_is_valid,
											num_subwindows);

		CompactArrays(num_subwindows);

//		HANDLE_ERROR(cudaMemcpy(&subwindows[0], dev_subwindows_in, sizeof(SubWindow) * num_subwindows, cudaMemcpyDeviceToHost));

//		subwindows.erase(remove_if(subwindows.begin(), subwindows.end(), isNonObject), subwindows.end());
//		DBG_WRP(cout << "Subwindows after stage " << i << " : " << subwindows.size() << endl << endl);

	}

	subwindows.resize(num_subwindows);
	HANDLE_ERROR(cudaMemcpy(&subwindows[0], dev_subwindows_in, sizeof(SubWindow) * num_subwindows, cudaMemcpyDeviceToHost));

}

GpuObjDetector::GpuObjDetector(int w, int h, HaarCascade& cascade) :
	img_width(w),
	img_height(h),
	haar_cascade(cascade) {

	img_mem_size = img_width * img_height * sizeof(int);
	ii_mem_size = (img_width + 1) * (img_height + 1) * sizeof(int);

	HANDLE_ERROR(cudaMalloc(&dev_img, img_mem_size));
	HANDLE_ERROR(cudaMalloc(&dev_ii, ii_mem_size));
	HANDLE_ERROR(cudaMalloc(&dev_ii2, ii_mem_size));

	PrecalcSubwindows(img_width,
					  img_height,
					  haar_cascade.window_width,
					  haar_cascade.window_height,
					  all_subwindows);

	HANDLE_ERROR(cudaMalloc(&dev_subwindows_in, sizeof(SubWindow) * all_subwindows.size()));
	HANDLE_ERROR(cudaMalloc(&dev_subwindows_out, sizeof(SubWindow) * all_subwindows.size()));
	HANDLE_ERROR(cudaMalloc(&dev_is_valid, sizeof(int) * all_subwindows.size()));
	HANDLE_ERROR(cudaMalloc(&dev_indexes, sizeof(int) * all_subwindows.size()));

	cudppCreate(&lib);
	scan_conf.op = CUDPP_ADD;
	scan_conf.datatype = CUDPP_INT;
	scan_conf.algorithm = CUDPP_SCAN;
	scan_conf.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

}


void GpuObjDetector::Detect(int *g_img, std::vector<SubWindow>& objs) {

	HANDLE_ERROR(cudaMemcpy(dev_img, g_img, img_mem_size, cudaMemcpyHostToDevice));
	GpuComputeII();
	objs = all_subwindows;
	DetectAtSubwindows(objs);
}


GpuObjDetector::~GpuObjDetector() {
	HANDLE_ERROR(cudaFree(dev_img));
	HANDLE_ERROR(cudaFree(dev_ii));
	HANDLE_ERROR(cudaFree(dev_ii2));
	HANDLE_ERROR(cudaFree(dev_subwindows_in));
	HANDLE_ERROR(cudaFree(dev_subwindows_out));
	HANDLE_ERROR(cudaFree(dev_is_valid));
	HANDLE_ERROR(cudaFree(dev_indexes));

}
