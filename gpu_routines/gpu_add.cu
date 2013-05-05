#include "gpu_add.h"
#include "gpu_utils.h"

__global__ void kernel_add(int* a, int* b, int* c, int* size) {
	int tid = blockIdx.x;
	if (tid < *size) {
		c[tid] = a[tid] + b[tid];
	}
}

void gpuVectorAdd(const int* a, const int* b, int* c, int size)
{
	int *dev_a, *dev_b, *dev_c, *dev_size;
	HANDLE_ERROR(cudaMalloc((void **)&dev_a, size * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_b, size * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_c, size * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_size, sizeof(int)));

	HANDLE_ERROR(cudaMemcpy((void *)dev_a, (void *)a, size * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy((void *)dev_b, (void *)b, size * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy((void *)dev_size, (void *)&size, sizeof(int), cudaMemcpyHostToDevice));

	kernel_add<<<size, 1>>>(dev_a, dev_b, dev_c, dev_size);

	HANDLE_ERROR(cudaMemcpy((void *)c, (void *)dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));
	HANDLE_ERROR(cudaFree(dev_size));
}
