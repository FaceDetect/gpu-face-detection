/*
 * gpu_utils.h
 *
 *  Created on: May 3, 2013
 *      Author: olehp
 */

#ifndef GPU_UTILS_H_
#define GPU_UTILS_H_

#define MATR_VAL(arr, r, c, w) \
	(( arr ) [ ( r ) * ( w ) + c ])

void HandleError(cudaError_t err, const char *file, int line);

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#endif /* GPU_UTILS_H_ */
