/*
 * gpu_utils.h
 *
 *  Created on: May 3, 2013
 *      Author: olehp
 */

#ifndef GPU_UTILS_H_
#define GPU_UTILS_H_

#include <cstdio>

static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#endif /* GPU_UTILS_H_ */
