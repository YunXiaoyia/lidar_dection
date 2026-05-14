#ifndef LIB_INFERENCE_TRT_PLUGINS_SCATTERMAXPLUGIN_SCATTERMAXKERNEL
#define LIB_INFERENCE_TRT_PLUGINS_SCATTERMAXPLUGIN_SCATTERMAXKERNEL

#include <algorithm>
#include <cassert>
#include <cstdio>
#include "cublas_v2.h"
#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;


void scatterMaxKernelLaunch(cudaStream_t stream, int batch_size, int max_num_points, int channels, int num_segments,
                            const float* src_data, const int* index, float* out_data);

#endif /* LIB_INFERENCE_TRT_PLUGINS_SCATTERMAXPLUGIN_SCATTERMAXKERNEL */
