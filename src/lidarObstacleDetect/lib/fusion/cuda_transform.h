#ifndef CUDA_TRANSFORM_H_
#define CUDA_TRANSFORM_H_

#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <ros/ros.h> // 用于ROS_ERROR

/**
 * @brief CUDA 错误检查宏 (CPU端)
 */
#define CUDA_CHECK(call)                                             \
{                                                                    \
    const cudaError_t error = call;                                  \
    if (error != cudaSuccess)                                        \
    {                                                                \
        ROS_ERROR("CUDA Error: %s:%d, ", __FILE__, __LINE__);        \
        ROS_ERROR("code: %d, reason: %s", error, cudaGetErrorString(error)); \
        exit(1);                                                     \
    }                                                                \
}

// +++ 滤波所需的数据结构 +++
struct ParamsBounds {
  float  x_min;
  float  x_max;
  float  y_min;
  float  y_max;
  float  z_min;
  float  z_max;
};

/**
 * @brief [异步 DtoD] 坐标变换核函数启动器
 *
 * @param d_in         [输入] 设备(GPU)上的输入点云
 * @param d_out        [输出] 设备(GPU)上的输出点云 (必须已分配)
 * @param d_matrix_row_major [输入] 设备(GPU)上的 4x4 行优先变换矩阵
 * @param num_points   点云中的总点数
 * @param stream       CUDA 流
 */
void transformPointCloudCUDA_async(
    const pcl::PointXYZI* d_in, 
    pcl::PointXYZI* d_out, 
    const float* d_matrix_row_major, 
    int num_points, 
    cudaStream_t stream
);

/**
 * @brief [异步 DtoD] 融合(拼接)多个GPU点云
 *
 * @param d_out           [输出] 融合后点云的设备指针 (必须已分配)
 * @param d_pc1, n_pc1    [输入] 点云1 (父)
 * @param d_pc2, n_pc2    [输入] 点云2 (子)
 * @param d_pc3, n_pc3    [输入] 点云3 (子)
 * @param d_pc4, n_pc4    [输入] 点云4 (子)
 * @param num_to_fuse     要融合的点云数量 (2, 3 或 4)
 * @param stream          CUDA 流
 */
void launch_fuse_clouds(pcl::PointXYZI* d_out,
                        const pcl::PointXYZI* d_pc1, size_t n_pc1,
                        const pcl::PointXYZI* d_pc2, size_t n_pc2,
                        const pcl::PointXYZI* d_pc3, size_t n_pc3,
                        const pcl::PointXYZI* d_pc4, size_t n_pc4,
                        int num_to_fuse,
                        cudaStream_t stream);


// +++ GPU滤波核函数启动器 +++
/**
 * @brief [异步 DtoD] 流式压缩滤波核函数启动器
 *
 * @param d_in            [输入] 输入点云的设备指针
 * @param n_in            [输入] 输入点云的点数
 * @param d_out           [输出] 输出点云的设备指针 (必须已分配，大小 >= n_in)
 * @param d_out_index     [输入/输出] 指向设备上单个uint的指针，用作原子计数器。必须用 cudaMemsetAsync 清零
 * @param bounds          [输入] 滤波边界参数
 * @param filter_inside   [输入] 模式。
 * true:  滤除边界内的点 (用于 internal_bounds)
 * false: 滤除边界外的点 (用于 external_bounds)
 * @param stream          CUDA 流
 */
void launch_filter_points(const pcl::PointXYZI* d_in, size_t n_in,
                          pcl::PointXYZI* d_out,
                          unsigned int* d_out_index,
                          const ParamsBounds* d_bounds_list,
                          int num_bounds,
                          bool filter_inside,
                          cudaStream_t stream);

#endif // CUDA_TRANSFORM_H_
