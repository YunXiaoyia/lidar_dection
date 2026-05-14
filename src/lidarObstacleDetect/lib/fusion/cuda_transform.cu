#include "cuda_transform.h"
#include <cuda_runtime.h>
#include <iostream>

// ==========================================
// 1. 坐标变换部分
// ==========================================

__global__
void transformKernel(const pcl::PointXYZI* in_points, 
                     pcl::PointXYZI* out_points, 
                     const float* matrix_data, 
                     int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    const pcl::PointXYZI& p_in = in_points[idx];
    pcl::PointXYZI& p_out = out_points[idx];

    const float x = p_in.x;
    const float y = p_in.y;
    const float z = p_in.z;

    p_out.x = matrix_data[0] * x + matrix_data[1] * y + matrix_data[2] * z + matrix_data[3];
    p_out.y = matrix_data[4] * x + matrix_data[5] * y + matrix_data[6] * z + matrix_data[7];
    p_out.z = matrix_data[8] * x + matrix_data[9] * y + matrix_data[10] * z + matrix_data[11];
    p_out.intensity = p_in.intensity;
}

void transformPointCloudCUDA_async(
    const pcl::PointXYZI* d_in, 
    pcl::PointXYZI* d_out, 
    const float* d_matrix_row_major, 
    int num_points, 
    cudaStream_t stream)
{
    if (num_points == 0) return;
    const int threads = 256;
    const int blocks = (num_points + threads - 1) / threads;
    transformKernel<<<blocks, threads, 0, stream>>>(d_in, d_out, d_matrix_row_major, num_points);
}

// ==========================================
// 2. 点云融合部分
// ==========================================

void launch_fuse_clouds(pcl::PointXYZI* d_out,
                        const pcl::PointXYZI* d_pc1, size_t n_pc1,
                        const pcl::PointXYZI* d_pc2, size_t n_pc2,
                        const pcl::PointXYZI* d_pc3, size_t n_pc3,
                        const pcl::PointXYZI* d_pc4, size_t n_pc4,
                        int num_to_fuse,
                        cudaStream_t stream)
{
    size_t offset_bytes = 0;
    size_t current_bytes = 0;
    
    // PC1
    current_bytes = n_pc1 * sizeof(pcl::PointXYZI);
    if (current_bytes > 0) {
        cudaMemcpyAsync((char*)d_out + offset_bytes, d_pc1, current_bytes, cudaMemcpyDeviceToDevice, stream);
        offset_bytes += current_bytes;
    }
    // PC2
    current_bytes = n_pc2 * sizeof(pcl::PointXYZI);
    if (current_bytes > 0) {
        cudaMemcpyAsync((char*)d_out + offset_bytes, d_pc2, current_bytes, cudaMemcpyDeviceToDevice, stream);
        offset_bytes += current_bytes;
    }
    // PC3
    current_bytes = n_pc3 * sizeof(pcl::PointXYZI);
    if (current_bytes > 0) {
        cudaMemcpyAsync((char*)d_out + offset_bytes, d_pc3, current_bytes, cudaMemcpyDeviceToDevice, stream);
        offset_bytes += current_bytes;
    }
    // PC4
    current_bytes = n_pc4 * sizeof(pcl::PointXYZI);
    if (current_bytes > 0) {
        cudaMemcpyAsync((char*)d_out + offset_bytes, d_pc4, current_bytes, cudaMemcpyDeviceToDevice, stream);
    }
}

// ==========================================
// 3. 边界滤波部分 (支持多框)
// ==========================================

/**
 * @brief 检查点是否在任意一个边界框内
 * 参数已更新为接收数组和数量
 */
__device__ __forceinline__ bool is_inside(const pcl::PointXYZI& p, 
                                          const ParamsBounds* bounds_list, 
                                          int num_bounds)
{
    for (int i = 0; i < num_bounds; ++i) {
        const ParamsBounds& b = bounds_list[i];
        if (p.x >= b.x_min && p.x <= b.x_max &&
            p.y >= b.y_min && p.y <= b.y_max &&
            p.z >= b.z_min && p.z <= b.z_max) 
        {
            return true; // 只要在一个盒子里，就返回 true
        }
    }
    return false;
}

__global__ void filter_kernel(const pcl::PointXYZI* d_in, size_t n_in,
                              pcl::PointXYZI* d_out,
                              unsigned int* d_out_index,
                              const ParamsBounds* d_bounds_list, // 这里改成了指针
                              int num_bounds,                    // 这里增加了数量
                              bool filter_inside)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_in) return;

    const pcl::PointXYZI p = d_in[idx];

    // 修复点：正确调用新的 is_inside，传入数组和数量
    bool point_is_inside = is_inside(p, d_bounds_list, num_bounds);
    bool keep_point = false;

    if (filter_inside) {
        // 模式：滤除内部点 (保留外部) -> 如车身过滤
        keep_point = !point_is_inside;
    } else {
        // 模式：滤除外部点 (保留内部) -> 如ROI过滤
        keep_point = point_is_inside;
    }

    if (keep_point)
    {
        unsigned int out_idx = atomicAdd(d_out_index, 1);
        d_out[out_idx] = p;
    }
}

void launch_filter_points(const pcl::PointXYZI* d_in, size_t n_in,
                          pcl::PointXYZI* d_out,
                          unsigned int* d_out_index,
                          const ParamsBounds* d_bounds_list, 
                          int num_bounds,
                          bool filter_inside,
                          cudaStream_t stream)
{
    if (n_in == 0) return;
    
    const int threads = 256;
    const int blocks = (n_in + threads - 1) / threads;

    filter_kernel<<<blocks, threads, 0, stream>>>(
        d_in, n_in, d_out, d_out_index, d_bounds_list, num_bounds, filter_inside
    );
}