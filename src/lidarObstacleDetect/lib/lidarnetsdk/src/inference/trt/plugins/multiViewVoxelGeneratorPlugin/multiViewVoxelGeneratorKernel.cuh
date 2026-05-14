
#ifndef LIB_INFERENCE_TRT_PLUGINS_MULTIVIEWVOXEL_GENERATOR_KERNEL
#define LIB_INFERENCE_TRT_PLUGINS_MULTIVIEWVOXEL_GENERATOR_KERNEL

#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

void generatorPointsCylinder_launch(int batch_size, int max_num_points, const float* points_xyz,
                                    const float* points_mask, float* points_cylinder, cudaStream_t stream);

void pointsToVoxels_launch(int batch_size, int max_num_points, int points_feat_num, int grid_size_x, int grid_size_y,
                           int grid_size_z, float voxel_size_x, float voxel_size_y, float voxel_size_z,
                           float min_range_x, float min_range_y, float min_range_z, float max_range_x,
                           float max_range_y, float max_range_z, const float* points, const float* points_mask,
                           float* voxel_coors, cudaStream_t stream);

void pointsStatus_launch(int batch_size, int max_num_points, int points_feat_num, int points_status_num,
                         int grid_size_x, int grid_size_y, int grid_size_z, float voxel_size_x, float voxel_size_y,
                         float voxel_size_z, float min_range_x, float min_range_y, float min_range_z,
                         const float* points, const float* points_mask, const float* voxel_coors, float* voxel_sum,
                         int* points_num_per_voxel, float* points_status, cudaStream_t stream);

#endif