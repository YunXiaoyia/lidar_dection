#ifndef LIB_INFERENCE_TRT_PLUGINS_PILLARSCATTERPLUGIN_PILLARSCATTERKERNEL
#define LIB_INFERENCE_TRT_PLUGINS_PILLARSCATTERPLUGIN_PILLARSCATTERKERNEL

#include <algorithm>
#include <cassert>
#include <cstdio>
#include "cublas_v2.h"
#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

template <typename Element>
int pillarScatterKernelLaunch(int batch_size, int max_pillar_num, int num_features, const Element *pillar_features_data,
                              const unsigned int *coords_data, const unsigned int *params_data, unsigned int featureX,
                              unsigned int featureY, Element *spatial_feature_data, cudaStream_t stream);

#endif /* LIB_INFERENCE_TRT_PLUGINS_PILLARSCATTERPLUGIN_PILLARSCATTERKERNEL */
