
#include "multiViewVoxelGeneratorKernel.cuh"
#include <iostream>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include "cublas_v2.h"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS


__global__ void generatorPointsCylider_kernel(int batch_size, int max_num_points, const int num_points_feat,
                                              const float* points_xyz, const float* points_mask,
                                              float* points_cylinder) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < max_num_points && points_mask[point_idx] > 0.5) {
        float x = points_xyz[num_points_feat * point_idx];
        float y = points_xyz[num_points_feat * point_idx + 1];
        float z = points_xyz[num_points_feat * point_idx + 2];
        float bev_dist = x * x + y * y;
        // float points_rho = sqrtf(bev_dist);
        // float points_phi = atan2f(y, x);
        points_cylinder[num_points_feat * point_idx] = atan2f(y, x);
        points_cylinder[num_points_feat * point_idx + 1] = z;
        points_cylinder[num_points_feat * point_idx + 2] = sqrtf(bev_dist);
    }
}

void generatorPointsCylinder_launch(int batch_size, int max_num_points, const float* points_xyz,
                                    const float* points_mask, float* points_cylinder, cudaStream_t stream) {
    assert(batch_size == 1);

    int threads = THREADS;
    int blocks = BLOCKS(max_num_points);
    generatorPointsCylider_kernel<<<blocks, threads, 0, stream>>>(batch_size, max_num_points, 3, points_xyz,
                                                                  points_mask, points_cylinder);
}


__global__ void pointsToVoxels_kernel(int batch_size, int max_num_points, int num_points_feat, int grid_size_x,
                                      int grid_size_y, int grid_size_z, float voxel_size_x, float voxel_size_y,
                                      float voxel_size_z, float min_range_x, float min_range_y, float min_range_z,
                                      float max_range_x, float max_range_y, float max_range_z, const float* points,
                                      const float* points_mask, float* voxel_coors) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < max_num_points && points_mask[point_idx] > 0.5) {
        float x = points[num_points_feat * point_idx];
        float y = points[num_points_feat * point_idx + 1];
        float z = points[num_points_feat * point_idx + 2];

        float coorx = (x - min_range_x) / voxel_size_x;
        float coory = (y - min_range_y) / voxel_size_y;
        float coorz = (z - min_range_z) / voxel_size_z;

        voxel_coors[num_points_feat * point_idx] = coorx;
        voxel_coors[num_points_feat * point_idx + 1] = coory;
        voxel_coors[num_points_feat * point_idx + 2] = coorz;
    }
}


void pointsToVoxels_launch(int batch_size, int max_num_points, int points_feat_num, int grid_size_x, int grid_size_y,
                           int grid_size_z, float voxel_size_x, float voxel_size_y, float voxel_size_z,
                           float min_range_x, float min_range_y, float min_range_z, float max_range_x,
                           float max_range_y, float max_range_z, const float* points, const float* points_mask,
                           float* voxel_coors, cudaStream_t stream) {
    assert(batch_size == 1);

    int threads = THREADS;
    int blocks = BLOCKS(max_num_points);
    pointsToVoxels_kernel<<<blocks, threads, 0, stream>>>(
        batch_size, max_num_points, points_feat_num, grid_size_x, grid_size_y, grid_size_z, voxel_size_x, voxel_size_y,
        voxel_size_z, min_range_x, min_range_y, min_range_z, max_range_x, max_range_y, max_range_z, points, points_mask,
        voxel_coors);
}


__global__ void scatterSum_kernel(int batch_size, int max_num_points, int num_points_feat, int grid_size_x,
                                  int grid_size_y, int grid_size_z, const float* points, const float* points_mask,
                                  const float* voxel_coors, float* voxel_sum, int* points_num_per_voxel) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx < max_num_points && points_mask[point_idx] > 0.5) {
        int point_idx_xyz = point_idx * num_points_feat;

        int voxel_x = int(voxel_coors[point_idx_xyz]);
        int voxel_y = int(voxel_coors[point_idx_xyz + 1]);
        // int voxel_z = int(voxel_coors[point_idx_xyz + 2]);

        int voxel_idx = (voxel_x * grid_size_y + voxel_y);
        int voxel_idx_xyz = voxel_idx * num_points_feat;

        assert(voxel_idx < grid_size_x * grid_size_y);

        atomicAdd(&voxel_sum[voxel_idx_xyz], points[point_idx_xyz]);
        atomicAdd(&voxel_sum[voxel_idx_xyz + 1], points[point_idx_xyz + 1]);
        atomicAdd(&voxel_sum[voxel_idx_xyz + 2], points[point_idx_xyz + 2]);

        atomicAdd(&points_num_per_voxel[voxel_idx], 1);
    }
}


// statue_feat_num = 9
__global__ void pointsStatus_Kernel(int batch_size, int max_num_points, int num_points_feat, int statue_feat_num,
                                    int grid_size_x, int grid_size_y, int grid_size_z, float voxel_size_x,
                                    float voxel_size_y, float voxel_size_z, float min_range_x, float min_range_y,
                                    float min_range_z, const float* points, const float* points_mask,
                                    const float* voxel_coors, const float* voxel_sum, const int* points_num_per_voxel,
                                    float* points_status) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < max_num_points && points_mask[point_idx] > 0.5) {
        int point_idx_xyz = point_idx * num_points_feat;
        int point_statue_idx = point_idx * statue_feat_num;
        // points_xyz
        points_status[point_statue_idx] = points[point_idx_xyz];
        points_status[point_statue_idx + 1] = points[point_idx_xyz + 1];
        points_status[point_statue_idx + 2] = points[point_idx_xyz + 2];

        // points_xyz_cluster
        int voxel_x = int(voxel_coors[point_idx_xyz]);
        int voxel_y = int(voxel_coors[point_idx_xyz + 1]);
        int voxel_z = int(voxel_coors[point_idx_xyz + 2]);

        int voxel_idx = voxel_x * grid_size_y + voxel_y;
        int voxel_idx_xyz = voxel_idx * num_points_feat;

        points_status[point_statue_idx + 3] =
            points[point_idx_xyz] - voxel_sum[voxel_idx_xyz] / points_num_per_voxel[voxel_idx];
        points_status[point_statue_idx + 4] =
            points[point_idx_xyz + 1] - voxel_sum[voxel_idx_xyz + 1] / points_num_per_voxel[voxel_idx];
        points_status[point_statue_idx + 5] =
            points[point_idx_xyz + 2] - voxel_sum[voxel_idx_xyz + 2] / points_num_per_voxel[voxel_idx];

        // f_center
        float voxel_center_x = (voxel_x + 0.5) * voxel_size_x + min_range_x;
        float voxel_center_y = (voxel_y + 0.5) * voxel_size_y + min_range_y;
        float voxel_center_z = (voxel_z + 0.5) * voxel_size_z + min_range_z;
        points_status[point_statue_idx + 6] = points[point_idx_xyz] - voxel_center_x;
        points_status[point_statue_idx + 7] = points[point_idx_xyz + 1] - voxel_center_y;
        points_status[point_statue_idx + 8] = points[point_idx_xyz + 2] - voxel_center_z;
    }
}

void pointsStatus_launch(int batch_size, int max_num_points, int points_feat_num, int points_status_num,
                         int grid_size_x, int grid_size_y, int grid_size_z, float voxel_size_x, float voxel_size_y,
                         float voxel_size_z, float min_range_x, float min_range_y, float min_range_z,
                         const float* points, const float* points_mask, const float* voxel_coors, float* voxel_sum,
                         int* points_num_per_voxel, float* points_status, cudaStream_t stream) {
    assert(batch_size == 1);

    int threads = THREADS;
    int blocks = BLOCKS(max_num_points);

    scatterSum_kernel<<<blocks, threads, 0, stream>>>(batch_size, max_num_points, points_feat_num, grid_size_x,
                                                      grid_size_y, grid_size_z, points, points_mask, voxel_coors,
                                                      voxel_sum, points_num_per_voxel);

    pointsStatus_Kernel<<<blocks, threads, 0, stream>>>(
        batch_size, max_num_points, points_feat_num, points_status_num, grid_size_x, grid_size_y, grid_size_z,
        voxel_size_x, voxel_size_y, voxel_size_z, min_range_x, min_range_y, min_range_z, points, points_mask,
        voxel_coors, voxel_sum, points_num_per_voxel, points_status);
}
