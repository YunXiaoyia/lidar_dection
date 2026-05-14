#ifndef LIB_INFERENCE_TRT_PLUGINS_SCATTERPLUGIN_SCATTERLAYER
#define LIB_INFERENCE_TRT_PLUGINS_SCATTERPLUGIN_SCATTERLAYER

#include "plugin.h"

pluginStatus_t scatterNDInference(cudaStream_t stream, int* outputDims, int nOutputDims, int sliceRank, int nRows,
    int rowSize, int CopySize, int sizeOfElementInBytes, const void* index,
    const void* updates, const void* data, void* output, void* workspace);


#endif /* LIB_INFERENCE_TRT_PLUGINS_SCATTERPLUGIN_SCATTERLAYER */
