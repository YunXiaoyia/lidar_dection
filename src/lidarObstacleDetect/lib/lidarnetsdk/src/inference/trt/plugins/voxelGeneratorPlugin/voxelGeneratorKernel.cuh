#ifndef LIB_INFERENCE_TRT_PLUGINS_VOXELGENERATORPLUGIN_VOXELGENERATORKERNEL
#define LIB_INFERENCE_TRT_PLUGINS_VOXELGENERATORPLUGIN_VOXELGENERATORKERNEL

#include "plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;

void generateVoxels_launch(int batch_size, int max_num_points, float *points, unsigned int *points_size,
                           float min_x_range, float max_x_range, float min_y_range, float max_y_range,
                           float min_z_range, float max_z_range, float pillar_x_size, float pillar_y_size,
                           float pillar_z_size, int grid_y_size, int grid_x_size, int num_point_values,
                           int max_points_per_voxel, unsigned int *mask, float *voxels, cudaStream_t stream);

void generateBaseFeatures_launch(int batch_size, unsigned int *mask, float *voxels, int grid_y_size, int grid_x_size,
                                 unsigned int *pillar_num, int max_pillar_num, int max_points_per_voxel,
                                 int num_point_values, float *voxel_features, unsigned int *voxel_num_points,
                                 unsigned int *coords, cudaStream_t stream);

int generateFeatures_launch(int batch_size, int dense_pillar_num, float *voxel_features, unsigned int *voxel_num_points,
                            unsigned int *coords, unsigned int *params, float voxel_x, float voxel_y, float voxel_z,
                            float range_min_x, float range_min_y, float range_min_z, unsigned int voxel_features_size,
                            unsigned int max_points, unsigned int max_voxels, unsigned int num_point_values,
                            float *features, cudaStream_t stream);

#endif /* LIB_INFERENCE_TRT_PLUGINS_VOXELGENERATORPLUGIN_VOXELGENERATORKERNEL */
