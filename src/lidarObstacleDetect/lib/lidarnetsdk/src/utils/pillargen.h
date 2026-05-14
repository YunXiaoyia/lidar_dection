// #pragma once

// #include <Eigen/Core>
// #include <Eigen/Dense>
// #include <Eigen/Geometry>
// #include <unsupported/Eigen/CXX11/Tensor>

// namespace lidar_net {

// // [新增] 1. 显式定义行优先矩阵类型 (对应 pillargen.cpp 中的实现)
// using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// using TensorMap = Eigen::TensorMap<Eigen::Tensor<float, 5, Eigen::RowMajor>>;  // waymo, nx5
// using TensorFMap = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>;
// using MatrixMap = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
// using MatrixIMap = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
// using VectorIMap = Eigen::Map<Eigen::VectorXi>;
// using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;

// /***************
// Inputs:
//     points                  : NxC, C: x, y, z, intensity, ...
//     voxel_size              :
//     pc_range_max            :
//     pc_range_min            :
//     max_points_per_pillar   :
//     max_voxels              :
//     voxel_feature_channel   :
// Outputs:
//     voxel_info              : shape=[max_voxels, max_points_per_pillar, voxel_feature_channel]
//     voxel_coors             : shape=[max_voxels, 4];  4: b, cz, cy, cx
//     voxel_num_points        : shape=[max_voxels]
//     voxel_valid_num         :

// *******************/

// // [新增] 2. 暴露具体的 Row 实现接口 (参数类型必须是 const RowMatrixXf&)
// // 这样你在回调函数里可以直接调用这个函数，或者让 VoxelGenerator 内部调用它而不发生隐式拷贝
// void VoxelGeneratorRow(const RowMatrixXf& points, const Eigen::Vector3f& voxel_size,
//                        const Eigen::Vector3f& pc_range_max, const Eigen::Vector3f& pc_range_min,
//                        const int max_points_per_pillar, const int max_voxels, const int voxel_feature_channel,
//                        TensorFMap& voxel_info, MatrixIMap& voxel_coors, VectorIMap& voxel_num_points,
//                        int& voxel_valid_num);

// // [新增] 3. 暴露具体的 Col 实现接口
// void VoxelGeneratorCol(const Eigen::MatrixXf& points, const Eigen::Vector3f& voxel_size,
//                        const Eigen::Vector3f& pc_range_max, const Eigen::Vector3f& pc_range_min,
//                        const int max_points_per_pillar, const int max_voxels, const int voxel_feature_channel,
//                        TensorFMap& voxel_info, MatrixIMap& voxel_coors, VectorIMap& voxel_num_points,
//                        int& voxel_valid_num);

// // [修改] 4. 主接口
// // 建议改为 Eigen::Ref 以避免隐式拷贝，或者保持原样但注意内部必须强制转换类型
// void VoxelGenerator(const Eigen::MatrixXf& points, const Eigen::Vector3f& voxel_size,
//                     const Eigen::Vector3f& pc_range_max, const Eigen::Vector3f& pc_range_min,
//                     const int max_points_per_pillar, const int max_voxels, const int voxel_feature_channel,
//                     TensorFMap& voxel_info, MatrixIMap& voxel_coors, VectorIMap& voxel_num_points, int& voxel_valid_num,
//                     const bool& col_major = true);

// }  // namespace lidar_net
#pragma once


#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/CXX11/Tensor>


namespace lidar_net {

using TensorMap = Eigen::TensorMap<Eigen::Tensor<float, 5, Eigen::RowMajor>>;  // waymo, nx5
using TensorFMap = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>;
using MatrixMap = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using MatrixIMap = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using VectorIMap = Eigen::Map<Eigen::VectorXi>;
using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;



/***************
Inputs:
    points                  : NxC, C: x, y, z, intensity, ...
    voxel_size              :
    pc_range_max            :
    pc_range_min            :
    max_points_per_pillar   :
    max_voxels              :
    voxel_feature_channel   :
Outputs:
    voxel_info              : shape=[max_voxels, max_points_per_pillar, voxel_feature_channel]
    voxel_coors             : shape=[max_voxels, 4];  4: b, cz, cy, cx
    voxel_num_points        : shape=[max_voxels]
    voxel_valid_num         :

*******************/

void VoxelGenerator(const Eigen::MatrixXf& points, const Eigen::Vector3f& voxel_size,
                    const Eigen::Vector3f& pc_range_max, const Eigen::Vector3f& pc_range_min,
                    const int max_points_per_pillar, const int max_voxels, const int voxel_feature_channel,
                    TensorFMap& voxel_info, MatrixIMap& voxel_coors, VectorIMap& voxel_num_points, int& voxel_valid_num,
                    const bool& col_major = true);





}  // namespace lidar_net
