#include "pillarScatter.h"
#include "pillarScatterCam.h"
#include "scatterPlugin.h"
#include "voxelGenerator.h"
#include "scatterMax.h"
#include "trt_grid_sampler.hpp"
#include "multiViewVoxelGenerator.h"

using mmdeploy::TRTGridSamplerCreator;

namespace nvinfer1 {

REGISTER_TENSORRT_PLUGIN(PillarScatterPluginCreator);
REGISTER_TENSORRT_PLUGIN(PillarScatterCamPluginCreator);
REGISTER_TENSORRT_PLUGIN(ScatterNDPluginCreator);
REGISTER_TENSORRT_PLUGIN(VoxelGeneratorPluginCreator);
REGISTER_TENSORRT_PLUGIN(ScatterMaxPluginCreator);
REGISTER_TENSORRT_PLUGIN(MultiViewVoxelGeneratorPluginCreator);
REGISTER_TENSORRT_PLUGIN(TRTGridSamplerCreator);

}  // namespace nvinfer1
