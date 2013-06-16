/*
 * gpu_prepare_data_set.cpp
 *
 *  Created on: May 15, 2013
 *      Author: olehp
 */

#include "gpu_prepare_data_set.h"
#include "gpu_utils.h"

#define MAX_THREADS_TO_PROCESS_DATA_SET 512

using namespace std;
using namespace cv;


inline __device__ float sqr(const float &arg) {
	return arg * arg;
}

inline __device__ __host__ float& ival(float *p, int row, int col) {
	return p[row * W_WIDTH + col];
}

inline __device__ __host__ float& iival(float *p, int row, int col) {
	return p[row * (W_WIDTH + 1) + col];
}

__device__ float eval_feature(const Feature& f, const float *ii) {

	float sum = 0;

	for (int i = 0; i < HAAR_MAX_RECTS; i++) {
		if (f.rects[i].wg == 0) continue;
		sum += ((ii[f.rects_coords[i].p0] +
				 ii[f.rects_coords[i].p3] -
				 ii[f.rects_coords[i].p2] -
				 ii[f.rects_coords[i].p1]) *
				 f.rects[i].wg);
	}

	return sum;
}

__device__ void to_ii(float *img, float *ii) {
	for (int y = 1; y < W_HEIGHT + 1; y++) {
		for (int x = 1; x < W_WIDTH + 1; x++) {

			float p4 = ival(img, y - 1, x - 1);
			float p3 = iival(ii, y, x - 1);
			float p2 = iival(ii, y - 1, x);
			float p1 = iival(ii, y - 1, x - 1);

			iival(ii, y, x) = p4 - p1 + p3 + p2;
		}
	}
}

__device__ void normalize_ii(float *img, float *ii) {
	float mean = iival(ii, W_HEIGHT, W_WIDTH) / IMG_SIZE;

	float sqr_sum = 0;

	for (int y = 0; y < W_HEIGHT; y++)
		for (int x = 0; x < W_WIDTH; x++)
			sqr_sum += sqr(ival(img, y, x));

	float variance = sqr_sum / IMG_SIZE - sqr(mean);

	float std_dev = (variance < 0) ? sqrt(-variance) : sqrt(variance);

	if (std_dev != 0)
		for (int y = 0; y < W_HEIGHT + 1; y++)
			for (int x = 0; x < W_WIDTH + 1; x++)
				iival(ii, y, x) /= std_dev;
}

__global__ void kernel_calc_iis_and_normalize(float *imgs,
										float *results,
										int num_imgs) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (!(i < num_imgs)) return;

	float *img = imgs + i * IMG_SIZE;
	float *result = results + i * II_SIZE;

	to_ii(img, result);
	normalize_ii(img, result);

}

__global__ void kernel_eval_features(float *feature_vals,
									 float *iis,
									 int i_img,
									 Feature *features,
									 int num_features) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (!(i < num_features)) return;

	float *ii = iis + i_img * II_SIZE;

	feature_vals[i] = eval_feature(features[i], ii);

}


void eval_all_features(Feature *dev_features, float *dev_results, int num_imgs, int num_features, Data& data) {

	int num_blocks = ceilf((float) num_features / MAX_THREADS_TO_PROCESS_DATA_SET);

	float *dev_feature_vals;
	HANDLE_ERROR(cudaMalloc(&dev_feature_vals, sizeof(float) * num_features));

	for (int i = 0; i < num_imgs; i++) {

		kernel_eval_features<<<num_blocks, MAX_THREADS_TO_PROCESS_DATA_SET>>>(dev_feature_vals,
																			  dev_results,
																			  i,
																			  dev_features,
																			  num_features);

		HANDLE_ERROR(cudaMemcpy(data.ptr<float>(i), dev_feature_vals, sizeof(float) * num_features, cudaMemcpyDeviceToHost));
	}

	HANDLE_ERROR(cudaFree(dev_feature_vals));

}



void imgs_vector_to_pointer(const std::vector<LabeledImg>& imgs, float *p_imgs, cv::Mat_<label_t>& labels) {

	for (int i = 0; i < imgs.size(); i++) {
		const float *data = imgs[i].first.ptr<float>(0);
		memcpy(p_imgs + IMG_SIZE * i, data, sizeof(float) * IMG_SIZE);
		labels(i, 0) = imgs[i].second;
	}
}

void pointer_to_imgs_vector(std::vector<LabeledImg>& imgs, float *results) {

	for (int i = 0; i < imgs.size(); i++) {
		imgs[i].first = Mat_<float>(W_HEIGHT + 1, W_WIDTH + 1, results + II_SIZE * i);

	}
}


void gpu_prepare_data_set(std::vector<LabeledImg>& imgs,
						  const std::vector<Feature>& feature_set,
						  cv::Mat_<label_t>& labels,
						  Data& data) {

	int num_imgs = imgs.size();
	int num_features = feature_set.size();

	float *p_imgs = new float[IMG_SIZE * imgs.size()];
	imgs_vector_to_pointer(imgs, p_imgs, labels);

	Feature *dev_feature_set;
	float *dev_imgs;
	float *dev_results;

	HANDLE_ERROR(cudaMalloc(&dev_feature_set, sizeof(Feature) * num_features));
	HANDLE_ERROR(cudaMemcpy(dev_feature_set, &feature_set[0], sizeof(Feature) * num_features, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc(&dev_imgs, sizeof(float) * IMG_SIZE * num_imgs));
	HANDLE_ERROR(cudaMemcpy(dev_imgs, p_imgs, sizeof(float) * IMG_SIZE * num_imgs, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc(&dev_results, sizeof(float) * II_SIZE * num_imgs));
	HANDLE_ERROR(cudaMemset(dev_results, 0, sizeof(float) * II_SIZE * num_imgs));


	int num_blocks = ceilf((float) num_imgs / MAX_THREADS_TO_PROCESS_DATA_SET);

	kernel_calc_iis_and_normalize<<<num_blocks, MAX_THREADS_TO_PROCESS_DATA_SET>>>(dev_imgs,
																			       dev_results,
																			       num_imgs);


	eval_all_features(dev_feature_set, dev_results, num_imgs, num_features, data);

	HANDLE_ERROR(cudaFree(dev_feature_set));
	HANDLE_ERROR(cudaFree(dev_imgs));
	HANDLE_ERROR(cudaFree(dev_results));

	delete [] p_imgs;

}
