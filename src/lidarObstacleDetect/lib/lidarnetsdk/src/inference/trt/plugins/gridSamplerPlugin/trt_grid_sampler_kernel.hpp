// Copyright (c) OpenMMLab. All rights reserved.
#ifndef SRC_INFERENCE_TRT_PLUGINS_GRIDSAMPLERPLUGIN_TRT_GRID_SAMPLER_KERNEL
#define SRC_INFERENCE_TRT_PLUGINS_GRIDSAMPLERPLUGIN_TRT_GRID_SAMPLER_KERNEL
#include <cuda_runtime.h>

enum class GridSamplerInterpolation { Bilinear, Nearest };
enum class GridSamplerPadding { Zeros, Border, Reflection };

template <typename T>
void grid_sample(T *output, const T *input, const T *grid, int *output_dims, int *input_dims, int *grid_dims,
                 int nb_dims, GridSamplerInterpolation interp, GridSamplerPadding padding, bool align_corners,
                 cudaStream_t stream);
#endif /* SRC_INFERENCE_TRT_PLUGINS_GRIDSAMPLERPLUGIN_TRT_GRID_SAMPLER_KERNEL */
