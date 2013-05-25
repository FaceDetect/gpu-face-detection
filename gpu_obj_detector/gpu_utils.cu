/*
 * gpu_utils.cu
 *
 *  Created on: May 3, 2013
 *      Author: olehp
 */

#include "gpu_utils.h"

#include <stdio.h>

void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
