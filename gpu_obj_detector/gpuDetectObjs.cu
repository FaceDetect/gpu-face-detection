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


#define SCALE_STEP 1.25
#define STEP_X 5
#define STEP_Y 5
#define MAX_SCALE 100
#define MAX_THREAD 256

__device__ inline int MatrVal(int *arr, int row, int col, int pic_width) {

	return arr[row * pic_width + col];
}

__device__ inline int RectSum(int* ii, int x, int y, int w, int h, int ii_width) {

	return MatrVal(ii, y, x, ii_width) +
		   MatrVal(ii, y + h, x + w, ii_width) -
		   MatrVal(ii, y, x + w, ii_width) -
		   MatrVal(ii, y + h, x, ii_width);
}

__global__ void kernel_detect_objs(int *ii,
								   int *ii2,
								   int ii_width,
								   int ii_height,
								   int width,
								   int height,
								   float scale,
								   int4 *stages,
								   int4 *features,
								   int4 *rects,
								   float *wgs,
								   float *num_objs) {
	// 244 216 123 123 6.19174

	*num_objs = 1;

	int x = blockIdx.x;
	int y = blockIdx.y;
	int4 window = {x, y, width, height};

	float inv = 1.0 / (height * width);
	float mean = RectSum(ii, x, y, width, height, ii_width) * inv;
	float variance = abs(RectSum(ii2, x, y, width, height, ii_width) * inv - OR_SQR(mean));

	float std_dev = sqrt(variance);

	for (int i = 0; i < HAAR_MAX_STAGES; i++) {
		int4 stage = stages[i];
		float stage_thr = __int_as_float(stage.x);
		float tree_sum = 0;
		short flen = (stage.w >> 16) & 0xffff;
		short fid = stage.w & 0xffff;

	    for (short j = fid; j < fid + flen; j++) {
	    	int4 feature = features[j];


	        float rects_sum = 0;//RectsPass(tree, x, y, scale) * inv;

	        short rlen = (feature.w >> 16) & 0xffff;
	        short rid = feature.w & 0xffff;
	        float f_thr = __int_as_float(feature.x);
	        float f_lval = __int_as_float(feature.y);
	        float f_rval = __int_as_float(feature.z);

	        for (short k = rid; k < rid + rlen; k++) {
				int4 rect = rects[k];

				rects_sum = rects_sum + RectSum(ii, x + (int)(rect.w * scale),
													y + (int)(rect.x * scale),
													(int)(rect.y * scale),
													(int)(rect.z * scale), ii_width) * wgs[k];
			}

	        tree_sum += ((rects_sum * inv < f_thr * std_dev) ? f_lval : f_rval);
	    }



		if (tree_sum < stage_thr) {
			return;
		}
	}
}

void detectAtScales(int *dev_ii,
					int* dev_ii2,
					int ii_width,
					int ii_height,
					float *dev_num_objs,
					int4 *dev_stages,
					int4 *dev_features,
					int4 *dev_rects,
					float *dev_wgs,
					int haar_window_width,
					int haar_window_height) {


	for (float scale = 1; scale < 1.1; scale *= SCALE_STEP) {
		int window_width = haar_window_width * scale;
		int window_height = haar_window_height * scale;

		dim3 grid;
		grid.x = ceilf((float)(ii_width - window_width) / STEP_X);
		grid.y = ceilf((float)(ii_height - window_height) / STEP_Y);
		grid.z = 1;

		dim3 block;
		block.x = 1;
		block.y = 1;
		block.z = 1;


		kernel_detect_objs<<<grid, block>>>(dev_ii,
											dev_ii2,
											ii_width,
											ii_height,
											window_width,
											window_height,
											scale,
											dev_stages,
											dev_features,
											dev_rects,
											dev_wgs,
											dev_num_objs);
	}


}


void gpuDetectObjs(cv::Mat_<int> img, HaarCascade& haar_cascade) {
	int img_width = img.rows;
	int img_height = img.cols;
//	int img_size = img_height * img_width;
	int ii_size = (img_height + 1) * (img_width + 1);

	int *ii = new int[ii_size];
	int *ii2 = new int[ii_size];

	float num_objs = 0;
	int4 *stages;
	int4 *features;
	int4 *rects;
	float *wgs;
	int num_stages;
	int num_features;
	int num_rects;

	HaarCascadeToArrays(haar_cascade, &stages, &features, &rects, &wgs, &num_stages, &num_features, &num_rects);

	float *dev_num_objs;
	int *dev_ii;
	int *dev_ii2;
	int4 *dev_stages;
	int4 *dev_features;
	int4 *dev_rects;
	float *dev_wgs;


	ComputeIIs(img.ptr<int>(), ii, ii2, img_width);

	HANDLE_ERROR(cudaMalloc((void **)&dev_stages, sizeof(int4) * num_stages));
	HANDLE_ERROR(cudaMalloc((void **)&dev_features, sizeof(int4) * num_features));
	HANDLE_ERROR(cudaMalloc((void **)&dev_rects, sizeof(int4) * num_rects));
	HANDLE_ERROR(cudaMalloc((void **)&dev_wgs, sizeof(float) * num_rects));

	HANDLE_ERROR(cudaMemcpy((void *)dev_stages, (void *)stages, sizeof(int4) * num_stages, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy((void *)dev_features, (void *)features, sizeof(int4) * num_features, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy((void *)dev_rects, (void *)rects, sizeof(int4) * num_rects, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy((void *)dev_wgs, (void *)wgs, sizeof(float) * num_rects, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void **)&dev_ii, sizeof(int) * ii_size));
	HANDLE_ERROR(cudaMalloc((void **)&dev_ii2, sizeof(int) * ii_size));
	HANDLE_ERROR(cudaMemcpy((void *)dev_ii, (void *)ii, sizeof(int) * ii_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy((void *)dev_ii2, (void *)ii2, sizeof(int) * ii_size, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void **)&dev_num_objs, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy((void *)dev_num_objs, (void *)&num_objs, sizeof(float), cudaMemcpyHostToDevice));




//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start, 0);

	detectAtScales(dev_ii,
				   dev_ii2,
				   img_width + 1,
				   img_height + 1,
				   dev_num_objs,
				   dev_stages,
				   dev_features,
				   dev_rects,
				   dev_wgs,
				   haar_cascade.window_width,
				   haar_cascade.window_height);

	HANDLE_ERROR(cudaDeviceSynchronize());

//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//
//	float elapsed;
//	cudaEventElapsedTime(&elapsed, start, stop);
//	cout << "Total elapsed: " << elapsed << endl;

	cudaMemcpy((void *)&num_objs, (void *)dev_num_objs, sizeof(float), cudaMemcpyDeviceToHost);

	cout << "Detected objs: " << num_objs << endl;


	HANDLE_ERROR(cudaFree(dev_ii));
	HANDLE_ERROR(cudaFree(dev_ii2));
	HANDLE_ERROR(cudaFree(dev_num_objs));
	HANDLE_ERROR(cudaFree(dev_stages));
	HANDLE_ERROR(cudaFree(dev_features));
	HANDLE_ERROR(cudaFree(dev_rects));
	HANDLE_ERROR(cudaFree(dev_wgs));


//	cout << "After gpuDetectObjs" << endl;

}













































//void PrecalcSubwindows(int img_width, int img_height, vector<SubWindow>& subwindows, HaarCascade& haar_cascade) {
//
//	float scale = 1.0;
//
//	int width = haar_cascade.window_width;
//	int height = haar_cascade.window_height;
//
//	while (OR_MIN(width, height) <= OR_MIN(img_width, img_height)) {
//
//		int x_step = 5;//OR_MAX(1, OR_MIN(4, floor(width / 10)));
//		int y_step = 5;//OR_MAX(1, OR_MIN(4, floor(height / 10)));
//
//		for (int y = 0; y < img_width - height; y += y_step) {
//			for (int x = 0; x < img_width - width; x += x_step) {
//				subwindows.push_back(SubWindow(x, y, width, height, scale));
//			}
//		}
//
//		scale = scale * 1.2;
//		width = (int)(haar_cascade.window_width * scale);
//		height = (int)(haar_cascade.window_height * scale);
//	}
//}








//bool isNonObject(const SubWindow& s) {
//	return !s.is_object;
//}


//void detectAtSubwindows(int *dev_ii, int *dev_ii2,
//						int img_width, int img_height,
//						HaarCascade *dev_haar_cascade,
//						float * dev_num_objs,
//						vector<SubWindow>& subwindows) {
//	float elapsed = 0;
//	for (int i = 0; i < HAAR_MAX_STAGES; i++) {
//
//		int num_subwindows = subwindows.size();
//		int num_blocks = ceilf((float) num_subwindows / MAX_THREAD);
//
//		SubWindow *dev_subwindows;
//		HANDLE_ERROR(cudaMalloc((void **)&dev_subwindows, sizeof(SubWindow) * num_subwindows));
//		HANDLE_ERROR(cudaMemcpy((void *)dev_subwindows, (void *)&subwindows[0], sizeof(SubWindow) * num_subwindows, cudaMemcpyHostToDevice));
//
//
//		cudaEvent_t start, stop;
//		cudaEventCreate(&start);
//		cudaEventCreate(&stop);
//		cudaEventRecord(start, 0);
//
//		kernel_detect_objs<<<num_blocks, MAX_THREAD>>>(i,
//											dev_ii,
//											dev_ii2,
//											img_width + 1,
//											img_height + 1,
//											dev_haar_cascade,
//											dev_subwindows,
//											subwindows.size(),
//											dev_num_objs);
//
//		cudaEventRecord(stop, 0);
//		cudaEventSynchronize(stop);
//
//		float tmp_elapsed;
//		cudaEventElapsedTime(&tmp_elapsed, start, stop);
//
//		elapsed += tmp_elapsed;
////		cout << "Elapsed by stage " << tmp_elapsed << endl;
//		HANDLE_ERROR(cudaMemcpy((void *)&subwindows[0], (void *)dev_subwindows, sizeof(SubWindow) * num_subwindows, cudaMemcpyDeviceToHost));
//
//		subwindows.erase(remove_if(subwindows.begin(), subwindows.end(), isNonObject), subwindows.end());
////		cout << "Subwindows after stage " << i << " : " << subwindows.size() << endl << endl;
//
//		HANDLE_ERROR(cudaFree(dev_subwindows));
//	}
//
////	cout << "Kernel elapsed: " << elapsed << endl;
//
//}







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



