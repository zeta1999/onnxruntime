// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param device_id cuda device id, starts from zero.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id);

/**
 * \param device_id cuda device id, starts from zero. 
 * \remarks Does not use an arena for CPU or GPU memory allocations made by the execution provider.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CUDA_NoArena, _In_ OrtSessionOptions* options, int device_id);

#ifdef __cplusplus
}
#endif
