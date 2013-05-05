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
#include <math.h>
#include <iostream>

using namespace std;

#define RECT_SUM(ii, x, y, w, h, img_w) \
	(MATR_VAL((ii), (y) - 1, (x) - 1, (img_w)) + \
	MATR_VAL((ii), (y) + (h) - 1, (x) + (w) - 1, (img_w)) - \
	MATR_VAL((ii), (y) - 1, (x) + (w) - 1, (img_w)) - \
	MATR_VAL((ii), (y) + (h) - 1, (x) - 1, (img_w)))


#define SCALE_UPDATE 0.8
#define MAX_NUM_OBJS 100

texture<int> tex_ii;
texture<int> tex_ii2;


__device__ int MatrVal(int row, int col, int pic_width) {

	return ((row == -1) || (col == -1)) ? 0 : tex1Dfetch(tex_ii, row * pic_width + col);
}

__device__ int MatrVal2(int row, int col, int pic_width) {

	return ((row == -1) || (col == -1)) ? 0 : tex1Dfetch(tex_ii2, row * pic_width + col);
}

__device__ int RectSum(int x, int y, int w, int h, int pic_width) {

	return MatrVal(y - 1, x - 1, pic_width) +
		   MatrVal(y + h - 1, x + w - 1, pic_width) -
		   MatrVal(y - 1, x + w - 1, pic_width) -
		   MatrVal(y + h - 1, x - 1, pic_width);
}

__device__ int RectSum2(int x, int y, int w, int h, int pic_width) {

	return MatrVal2(y - 1, x - 1, pic_width) +
		   MatrVal2(y + h - 1, x + w - 1, pic_width) -
		   MatrVal2(y - 1, x + w - 1, pic_width) -
		   MatrVal2(y + h - 1, x - 1, pic_width);
}

__global__ void kernel_detect_objs(int *ii,
								   int *ii2,
								   int img_width,
								   int img_height,
								   HaarCascade *haar_cascade,
								   Rectangle *objs,
								   float *num_objs) {
//	*num_objs = 123;
	// 244 216 123 123 6.19174
	int num_scale = blockIdx.y;
	float scale = __powf(1.0 / SCALE_UPDATE, num_scale);

	int width = haar_cascade->window_width * scale;
	int height = haar_cascade->window_height * scale;

	int x = threadIdx.x;
	int y = blockIdx.x;

	if (((x + width) > img_width) || ((y + height) > img_height)) return;

	float inv = 1.0 / (height * width);
	float mean = RectSum(x, y, width, height, img_width) * inv;
	float variance = abs(RectSum2(x, y, width, height, img_width) * inv - OR_SQR(mean));


	float std_dev = sqrt(variance);

	for (int i = 0; i < HAAR_MAX_STAGES; i++) {
		Stage &stage = haar_cascade->stages[i];
//		if (!stage.valid) break;

		float tree_sum = 0;//TreesPass(stage, x, y, scale, inv, std_dev);


	    for (int j = 0; j < HAAR_MAX_TREES; j++) {
	    	Tree& tree = stage.trees[j];
	    	if (!tree.valid) break;

	        float rects_sum = 0;//RectsPass(tree, x, y, scale) * inv;

	        for (int k = 0; k < HAAR_MAX_RECTS; k++) {
				Rectangle &rect = tree.feature.rects[k];
				if (rect.wg == 0) break;

				rects_sum = rects_sum + RectSum(x + (int)(rect.x * scale),
												y + (int)(rect.y * scale),
												(int)(rect.w * scale),
												(int)(rect.h * scale), img_width) * rect.wg;
			}

	        tree_sum += ((rects_sum * inv < tree.threshold * std_dev) ? tree.left_val : tree.right_val);
	    }



		if (tree_sum < stage.threshold) {
			return;
		}
	}
	*num_objs = 1;

}


void gpuDetectObjs(cv::Mat_<int> img, HaarCascade& haar_cascade) {
	int img_width = img.rows;
	int img_height = img.cols;

	float scale_width = (float)img_width / haar_cascade.window_width;
	float scale_height = (float)img_height / haar_cascade.window_height;
	float start_scale = OR_MIN(scale_width, scale_height);

	int num_scales = ceilf(log(1.0 / start_scale) / log(SCALE_UPDATE));

	int max_num_patches_y = img_height - haar_cascade.window_height;
	int max_num_patches_x = img_width - haar_cascade.window_width;

	int *ii = new int[img_height * img_width];
	int *ii2 = new int[img_height * img_width];


	int max_num_objects = MAX_NUM_OBJS;

	float num_objs = 0;
	Rectangle *objects = new Rectangle[max_num_objects];

	Rectangle *dev_objects;
	float *dev_num_objs;
	int *dev_ii;
	int *dev_ii2;
	HaarCascade *dev_haar_cascade;

	cout << "Image size = " << img_width << " x " << img_height << endl;

	gpuComputeII(img.ptr<int>(), ii, ii2, img_width, img_height);

	cout << "Start scale: " << start_scale << endl;
	cout << "Num scales: " << num_scales << endl;
	cout << "Max num patches X: " << max_num_patches_x << endl;
	cout << "Max num patches Y: " << max_num_patches_y << endl;



	HANDLE_ERROR(cudaMalloc((void **)&dev_haar_cascade, sizeof(haar_cascade)));
	HANDLE_ERROR(cudaMemcpy((void *)dev_haar_cascade, (void *)&haar_cascade, sizeof(haar_cascade), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void **)&dev_ii, sizeof(int) * img_width * img_height));
	HANDLE_ERROR(cudaMalloc((void **)&dev_ii2, sizeof(int) * img_width * img_height));
	HANDLE_ERROR(cudaMemcpy((void *)dev_ii, (void *)ii, sizeof(int) * img_width * img_height, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy((void *)dev_ii2, (void *)ii2, sizeof(int) * img_width * img_height, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaBindTexture(NULL, tex_ii, dev_ii, sizeof(int) * img_width * img_height));
	HANDLE_ERROR(cudaBindTexture(NULL, tex_ii2, dev_ii2, sizeof(int) * img_width * img_height));


	HANDLE_ERROR(cudaMalloc((void **)&dev_objects, sizeof(Rectangle) * max_num_objects));
	HANDLE_ERROR(cudaMemcpy((void *)dev_objects, (void *)objects, sizeof(Rectangle) * max_num_objects, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void **)&dev_num_objs, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy((void *)dev_num_objs, (void *)&num_objs, sizeof(float), cudaMemcpyHostToDevice));

	dim3 grid;
	grid.y = num_scales;
	grid.x = max_num_patches_y;
	grid.z = 1;

	dim3 block(max_num_patches_x);
	block.x = max_num_patches_x;
	block.y = block.z = 1;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	kernel_detect_objs<<<100, 250>>>(dev_ii,
										dev_ii2,
										img_width,
										img_height,
										dev_haar_cascade,
										dev_objects,
										dev_num_objs);

	HANDLE_ERROR(cudaDeviceSynchronize());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	cout << "Time elapsed: " << elapsed << endl;

	cudaMemcpy((void *)&num_objs, (void *)dev_num_objs, sizeof(float), cudaMemcpyDeviceToHost);

	cout << "Detected objs: " << num_objs << endl;

	cudaMemcpy((void *)objects, (void *)dev_objects, sizeof(Rectangle) * max_num_objects, cudaMemcpyDeviceToHost);

	for (int i = 0; i < num_objs; i++) {
		//if (objects[i].wg == 0) break;

//		cout << objects[i].x << " " << objects[i].y << " " << objects[i].w << " " << objects[i].h << " ";
	}

	HANDLE_ERROR(cudaFree(dev_haar_cascade));
	HANDLE_ERROR(cudaFree(dev_ii));
	HANDLE_ERROR(cudaFree(dev_ii2));
	HANDLE_ERROR(cudaFree(dev_num_objs));
	HANDLE_ERROR(cudaFree(dev_objects));
	HANDLE_ERROR(cudaUnbindTexture(tex_ii));
	HANDLE_ERROR(cudaUnbindTexture(tex_ii2));


	cout << "After gpuDetectObjs" << endl;

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



