/*
 * gpuDetectObjs.cu
 *
 *  Created on: May 4, 2013
 *      Author: olehp
 */

#include "gpuDetectObjs.h"
#include "gpu_compute_ii.h"
#include "gpu_utils.h"

#include "utils.h"
#include "SubWindow.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

#define MAX_THREAD 256

__device__ inline int MatrVal(int *arr, int row, int col, int pic_width) {

	return ((row == -1) || (col == -1)) ? 0 : arr[row * pic_width + col];
}

__device__ inline int RectSum(int* ii, int x, int y, int w, int h, int pic_width) {

	return MatrVal(ii, y - 1, x - 1, pic_width) +
		   MatrVal(ii, y + h - 1, x + w - 1, pic_width) -
		   MatrVal(ii, y - 1, x + w - 1, pic_width) -
		   MatrVal(ii, y + h - 1, x - 1, pic_width);
}

__global__ void kernel_detect_objs(int num_stage,
								   int *ii,
								   int *ii2,
								   int img_width,
								   int img_height,
								   HaarCascade *haar_cascade,
								   SubWindow *subwindows,
								   int num_subwindows,
								   float *num_objs) {
//	*num_objs = 123;
	// 244 216 123 123 6.19174

	*num_objs = 1;

	int i_subwindow = threadIdx.x + blockIdx.x * blockDim.x;
//	subwindows[i_subwindow].is_object = 0;

	if (!(i_subwindow < num_subwindows)) return;

	float scale = subwindows[i_subwindow].scale;
	int x = subwindows[i_subwindow].x;
	int y = subwindows[i_subwindow].y;
	int width = subwindows[i_subwindow].w;
	int height = subwindows[i_subwindow].h;

	float inv = 1.0 / (height * width);
	float mean = RectSum(ii, x, y, width, height, img_width) * inv;
	float variance = abs(RectSum(ii2, x, y, width, height, img_width) * inv - OR_SQR(mean));

	float std_dev = sqrt(variance);

	Stage &stage = haar_cascade->stages[num_stage];

	float tree_sum = 0;//TreesPass(stage, x, y, scale, inv, std_dev);

	for (int j = 0; j < HAAR_MAX_TREES; j++) {
		Tree& tree = stage.trees[j];
		if (!tree.valid) break;

		float rects_sum = 0;//RectsPass(tree, x, y, scale) * inv;

		for (int k = 0; k < HAAR_MAX_RECTS; k++) {
			Rectangle &rect = tree.feature.rects[k];
			if (rect.wg == 0) break;

			rects_sum = rects_sum + RectSum(ii, x + (int)(rect.x * scale),
												y + (int)(rect.y * scale),
												(int)(rect.w * scale),
												(int)(rect.h * scale), img_width) * rect.wg;
		}

		tree_sum += ((rects_sum * inv < tree.threshold * std_dev) ? tree.left_val : tree.right_val);
	}



	if (tree_sum < stage.threshold) {
		subwindows[i_subwindow].is_object = 0;
	}

}


void PrecalcSubwindows(int img_width, int img_height, vector<SubWindow>& subwindows, HaarCascade& haar_cascade) {

	float scale = 1.0;

	int width = haar_cascade.window_width;
	int height = haar_cascade.window_height;

	while (OR_MIN(width, height) <= OR_MIN(img_width, img_height)) {

		int x_step = OR_MAX(1, OR_MIN(4, floor(width / 10)));
		int y_step = OR_MAX(1, OR_MIN(4, floor(height / 10)));

		for (int y = 0; y < img_width - height; y += y_step) {
			for (int x = 0; x < img_width - width; x += x_step) {
				subwindows.push_back(SubWindow(x, y, width, height, scale));
			}
		}

		scale = scale * 1.2;
		width = (int)(haar_cascade.window_width * scale);
		height = (int)(haar_cascade.window_height * scale);
	}
}

bool isNonObject(const SubWindow& s) {
	return !s.is_object;
}

void detectAtSubwindows(int *dev_ii, int *dev_ii2,
						int img_width, int img_height,
						HaarCascade *dev_haar_cascade,
						float * dev_num_objs,
						vector<SubWindow>& subwindows) {
	float elapsed = 0;
	for (int i = 0; i < HAAR_MAX_STAGES; i++) {

		int num_subwindows = subwindows.size();
		int num_blocks = ceilf((float) num_subwindows / MAX_THREAD);

		SubWindow *dev_subwindows;
		HANDLE_ERROR(cudaMalloc((void **)&dev_subwindows, sizeof(SubWindow) * num_subwindows));
		HANDLE_ERROR(cudaMemcpy((void *)dev_subwindows, (void *)&subwindows[0], sizeof(SubWindow) * num_subwindows, cudaMemcpyHostToDevice));


		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		kernel_detect_objs<<<num_blocks, MAX_THREAD>>>(i,
											dev_ii,
											dev_ii2,
											img_width,
											img_height,
											dev_haar_cascade,
											dev_subwindows,
											subwindows.size(),
											dev_num_objs);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		float tmp_elapsed;
		cudaEventElapsedTime(&tmp_elapsed, start, stop);

		elapsed += tmp_elapsed;
//		cout << "Elapsed by stage " << tmp_elapsed << endl;
		HANDLE_ERROR(cudaMemcpy((void *)&subwindows[0], (void *)dev_subwindows, sizeof(SubWindow) * num_subwindows, cudaMemcpyDeviceToHost));

		subwindows.erase(remove_if(subwindows.begin(), subwindows.end(), isNonObject), subwindows.end());
//		cout << "Subwindows after stage " << i << " : " << subwindows.size() << endl << endl;
		
		HANDLE_ERROR(cudaFree(dev_subwindows));
	}

//	cout << "Kernel elapsed: " << elapsed << endl;

}


void gpuDetectObjs(cv::Mat_<int> img, HaarCascade& haar_cascade) {
	int img_width = img.rows;
	int img_height = img.cols;

	int *ii = new int[img_height * img_width];
	int *ii2 = new int[img_height * img_width];

	vector<SubWindow> subwindows;
	PrecalcSubwindows(img_width, img_height, subwindows, haar_cascade);


	float num_objs = 0;
	float *dev_num_objs;
	int *dev_ii;
	int *dev_ii2;
	HaarCascade *dev_haar_cascade;
//	cout << "Subwindows count: " << subwindows.size() << endl;
//	cout << "Image size = " << img_width << " x " << img_height << endl;

	gpuComputeII(img.ptr<int>(), ii, ii2, img_width, img_height);

	HANDLE_ERROR(cudaMalloc((void **)&dev_haar_cascade, sizeof(haar_cascade)));
	HANDLE_ERROR(cudaMemcpy((void *)dev_haar_cascade, (void *)&haar_cascade, sizeof(haar_cascade), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void **)&dev_ii, sizeof(int) * img_width * img_height));
	HANDLE_ERROR(cudaMalloc((void **)&dev_ii2, sizeof(int) * img_width * img_height));
	HANDLE_ERROR(cudaMemcpy((void *)dev_ii, (void *)ii, sizeof(int) * img_width * img_height, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy((void *)dev_ii2, (void *)ii2, sizeof(int) * img_width * img_height, cudaMemcpyHostToDevice));


	HANDLE_ERROR(cudaMalloc((void **)&dev_num_objs, sizeof(float)));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	detectAtSubwindows(dev_ii, dev_ii2, img_width, img_height, dev_haar_cascade, dev_num_objs, subwindows);

	HANDLE_ERROR(cudaDeviceSynchronize());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
//	cout << "Total elapsed: " << elapsed << endl;

	cudaMemcpy((void *)&num_objs, (void *)dev_num_objs, sizeof(float), cudaMemcpyDeviceToHost);

//	cout << "Detected objs: " << num_objs << endl;


	for (int i = 0; i < subwindows.size(); i++) {

		cout << subwindows[i].x << " " << subwindows[i].y << " " << subwindows[i].w << " " << subwindows[i].h << " ";
	}

	HANDLE_ERROR(cudaFree(dev_haar_cascade));
	HANDLE_ERROR(cudaFree(dev_ii));
	HANDLE_ERROR(cudaFree(dev_ii2));
	HANDLE_ERROR(cudaFree(dev_num_objs));

//	cout << "After gpuDetectObjs" << endl;

}


















//__device__ float RectsPass(Tree &tree,
//						   int x,
//						   int y,
//						   int *ii,
//						   float scale,
//						   int img_width) {
//	float rects_sum = 0;
//	for (int i = 0; i < HAAR_MAX_RECTS; i++) {
//		Rectangle &rect = tree.feature.rects[i];
//		if (rect.wg == 0) break;
//
//		rects_sum = rects_sum +
//				RectSum(ii,
//						 x + (int)(rect.x * scale),
//						 y + (int)(rect.y * scale),
//						 (int)(rect.w * scale),
//						 (int)(rect.h * scale),
//						 img_width) * rect.wg;
//	}
//
//	return rects_sum;
//}
//
//__device__ float TreesPass(Stage &stage,
//		                   int x,
//		                   int y,
//		                   int *ii,
//		                   float scale,
//		                   float inv,
//		                   float std_dev,
//		                   int img_width) {
//
//    float tree_sum = 0;
//
//    for (int i = 0; i < HAAR_MAX_TREES; i++) {
//    	Tree& tree = stage.trees[i];
//    	if (!tree.valid) break;
//
//        float rects_sum = RectsPass(tree, x, y, ii, scale, img_width) * inv;
//
//        if (rects_sum < tree.threshold * std_dev)
//            tree_sum = tree_sum + tree.left_val;
//        else
//            tree_sum = tree_sum + tree.right_val;
//    }
//
//    return tree_sum;
//}
//
//__device__ bool StagesPass(int x,
//		                   int y,
//		                   int *ii,
//		                   float inv,
//		                   float std_dev,
//		                   float scale,
//		                   int img_width,
//		                   HaarCascade *haar_cascade) {
//
//	for (int i = 0; i < HAAR_MAX_STAGES; i++) {
//		Stage &stage = haar_cascade->stages[i];
//		if (!stage.valid) break;
//
//		float tree_sum = TreesPass(stage, x, y, ii, scale, inv, std_dev, img_width);
//		if (tree_sum < stage.threshold) {
//			return false;
//		}
//	}
//
//	return true;
//}












//bool gpuDetectObjsAt(int *ii,
//					 int *ii2,
//					 float scale,
//					 int x,
//					 int y,
//					 int width,
//					 int height,
//					 int img_width,
//					 int img_height,
//					 HaarCascade& haar_cascade) {
//
////	cout << "Starting gpuDetectObjsAt" << endl;
//
////	cout << "x = " << x << endl;
////	cout << "y = " << y << endl;
////	cout << "w = " << width << endl;
////	cout << "h = " << height << endl;
////	cout << "scale = " << scale << endl;
//
////	244 216 123 123
//	float result = 0;
//	HaarCascade *dev_haar_cascade;
//	int *dev_ii;
//	int *dev_ii2;
//	float *dev_result;
//
//
//	HANDLE_ERROR(cudaMalloc((void **)&dev_result, sizeof(float)));
//	HANDLE_ERROR(cudaMemcpy((void *)dev_result, (void *)&result, sizeof(float), cudaMemcpyHostToDevice));
//
//	HANDLE_ERROR(cudaMalloc((void **)&dev_haar_cascade, sizeof(haar_cascade)));
//	HANDLE_ERROR(cudaMemcpy((void *)dev_haar_cascade, (void *)&haar_cascade, sizeof(haar_cascade), cudaMemcpyHostToDevice));
//
//	HANDLE_ERROR(cudaMalloc((void **)&dev_ii, sizeof(int) * img_width * img_height));
//	HANDLE_ERROR(cudaMalloc((void **)&dev_ii2, sizeof(int) * img_width * img_height));
//	HANDLE_ERROR(cudaMemcpy((void *)dev_ii, (void *)ii, sizeof(int) * img_width * img_height, cudaMemcpyHostToDevice));
//	HANDLE_ERROR(cudaMemcpy((void *)dev_ii2, (void *)ii2, sizeof(int) * img_width * img_height, cudaMemcpyHostToDevice));
//
//
//	kernel_detect_objs<<<1, 1>>>(dev_ii, dev_ii2, x, y, width, height, img_width, img_height, scale, dev_haar_cascade, dev_result);
////
//	cudaMemcpy((void *)&result, (void *)dev_result, sizeof(int), cudaMemcpyDeviceToHost);
//
//	HANDLE_ERROR(cudaFree(dev_result));
//	HANDLE_ERROR(cudaFree(dev_haar_cascade));
//	HANDLE_ERROR(cudaFree(dev_ii));
//	HANDLE_ERROR(cudaFree(dev_ii2));
//
////	cout << "Exiting gpuDetectObjsAt" << endl;
//
//	cout << "Result: " << result << endl;
//	HANDLE_ERROR(cudaDeviceSynchronize());
//
//	return result;
//}



