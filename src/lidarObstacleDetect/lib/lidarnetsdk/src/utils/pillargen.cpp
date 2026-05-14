// #include <cmath>
// #include <iostream>
// #include "log/logging.h"
// #include "pillargen.h"
// #include "common/deepways_time.h"

// namespace lidar_net {
// using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// void VoxelGeneratorCol(const Eigen::MatrixXf& points, const Eigen::Vector3f& voxel_size,
//                        const Eigen::Vector3f& pc_range_max, const Eigen::Vector3f& pc_range_min,
//                        const int max_points_per_pillar, const int max_voxels, const int voxel_feature_channel,
//                        TensorFMap& voxel_info, MatrixIMap& voxel_coors, VectorIMap& voxel_num_points,
//                        int& voxel_valid_num) {
//     Eigen::VectorXi grid_size = ((pc_range_max - pc_range_min).array() / voxel_size.array()).floor().cast<int>();
//     Eigen::VectorXi coors_x = ((points.col(0).array() - pc_range_min(0)) / voxel_size(0)).floor().cast<int>();
//     Eigen::VectorXi coors_y = ((points.col(1).array() - pc_range_min(1)) / voxel_size(1)).floor().cast<int>();
//     Eigen::VectorXi coors_z = ((points.col(2).array() - pc_range_min(2)) / voxel_size(2)).floor().cast<int>();

//     // max_voxels
//     voxel_valid_num = 0;
//     Eigen::MatrixXi&& coor_to_voxelidx = Eigen::MatrixXi::Constant(grid_size(0), grid_size(1), -1);

//     int points_num = points.rows();
//     for (int i = 0; i < points_num; ++i) {
//         // TODO, mask
//         if (coors_x(i) < 0 || coors_x(i) >= grid_size.x() || coors_y(i) < 0 || coors_y(i) >= grid_size.y() ||
//             coors_z(i) < 0 || coors_z(i) >= grid_size.z()) {
//             continue;
//         }

//         if (voxel_valid_num >= max_voxels) {
//             ///
//             return;
//         }

//         int& pillar_index = coor_to_voxelidx(coors_x(i), coors_y(i));

//         if (pillar_index == -1) {
//             // new
//             pillar_index = voxel_valid_num;
//             voxel_info(pillar_index, voxel_num_points(pillar_index), 0) = points(i, 0);
//             voxel_info(pillar_index, voxel_num_points(pillar_index), 1) = points(i, 1);
//             voxel_info(pillar_index, voxel_num_points(pillar_index), 2) = points(i, 2);
//             voxel_info(pillar_index, voxel_num_points(pillar_index), 3) = points(i, 3);
//             // voxel_info(pillar_index, voxel_num_points(pillar_index), 4) = points( 4,i);

//             voxel_coors.row(pillar_index) << 0, 0, coors_y(i), coors_x(i);
//             voxel_num_points(pillar_index) += 1;
//             voxel_valid_num += 1;

//         } else {
//             int& num = voxel_num_points(pillar_index);
//             if (num < max_points_per_pillar) {
//                 voxel_info(pillar_index, voxel_num_points(pillar_index), 0) = points(i, 0);
//                 voxel_info(pillar_index, voxel_num_points(pillar_index), 1) = points(i, 1);
//                 voxel_info(pillar_index, voxel_num_points(pillar_index), 2) = points(i, 2);
//                 voxel_info(pillar_index, voxel_num_points(pillar_index), 3) = points(i, 3);
//                 // voxel_info(pillar_index, voxel_num_points(pillar_index), 4) = points( 4,i);
//                 num += 1;
//             }
//         }
//     }
// }

// // SOTA 优化版 VoxelGeneratorRow
// // points 是 Row-Major, N 行 x C 列 (C>=4)
// void VoxelGeneratorRow(const RowMatrixXf& points, const Eigen::Vector3f& voxel_size,
//                        const Eigen::Vector3f& pc_range_max, const Eigen::Vector3f& pc_range_min,
//                        const int max_points_per_pillar, const int max_voxels, const int voxel_feature_channel,
//                        TensorFMap& voxel_info, MatrixIMap& voxel_coors, VectorIMap& voxel_num_points,
//                        int& voxel_valid_num) {
    
//     // 1. 预计算常量
//     const float inv_voxel_x = 1.0f / voxel_size(0);
//     const float inv_voxel_y = 1.0f / voxel_size(1);
//     const float inv_voxel_z = 1.0f / voxel_size(2);
//     const float min_x_range = pc_range_min(0);
//     const float min_y_range = pc_range_min(1);
//     const float min_z_range = pc_range_min(2);
    
//     int grid_x_size = static_cast<int>((pc_range_max(0) - min_x_range) * inv_voxel_x);
//     int grid_y_size = static_cast<int>((pc_range_max(1) - min_y_range) * inv_voxel_y);
//     int grid_z_size = static_cast<int>((pc_range_max(2) - min_z_range) * inv_voxel_z);

//     // 2. 初始化 Grid Map
//     voxel_valid_num = 0;
//     Eigen::MatrixXi coor_to_voxelidx = Eigen::MatrixXi::Constant(grid_x_size, grid_y_size, -1);

//     // 3. 单次遍历
//     const int points_num = points.rows(); 
    
//     // 【验证】此时 points.cols() 应该是 4 或 5，且 IsRowMajor 必须为 true
//     // 如果这里 data 乱了，说明外面传入的数据类型还是不对
//     const int stride = points.cols(); 
//     const float* data_ptr = points.data(); 

//     for (int i = 0; i < points_num; ++i) {
//         // 利用行优先连续内存访问
//         const float* p_curr = data_ptr + i * stride;

//         float px = p_curr[0];
//         float py = p_curr[1];
//         float pz = p_curr[2];
//         // float pi = p_curr[3]; 

//         // 越界检查 (这里加上更严格的检查，防止无效点)
//         if (px < min_x_range || px >= pc_range_max(0) ||
//             py < min_y_range || py >= pc_range_max(1) ||
//             pz < min_z_range || pz >= pc_range_max(2)) {
//             continue;
//         }

//         int coor_x = static_cast<int>((px - min_x_range) * inv_voxel_x);
//         int coor_y = static_cast<int>((py - min_y_range) * inv_voxel_y);
//         // int coor_z = static_cast<int>((pz - min_z_range) * inv_voxel_z);

//         // 边界保护
//         if (coor_x < 0 || coor_x >= grid_x_size || coor_y < 0 || coor_y >= grid_y_size) continue;

//         int& pillar_idx = coor_to_voxelidx(coor_x, coor_y);

//         if (pillar_idx == -1) {
//             if (voxel_valid_num >= max_voxels) continue;

//             pillar_idx = voxel_valid_num;
//             voxel_valid_num++;
            
//             voxel_num_points(pillar_idx) = 0;
//             voxel_coors(pillar_idx, 0) = 0;
//             voxel_coors(pillar_idx, 1) = 0;
//             voxel_coors(pillar_idx, 2) = coor_y;
//             voxel_coors(pillar_idx, 3) = coor_x;
//         }

//         int& num_in_voxel = voxel_num_points(pillar_idx);
//         if (num_in_voxel < max_points_per_pillar) {
//             voxel_info(pillar_idx, num_in_voxel, 0) = px;
//             voxel_info(pillar_idx, num_in_voxel, 1) = py;
//             voxel_info(pillar_idx, num_in_voxel, 2) = pz;
//             voxel_info(pillar_idx, num_in_voxel, 3) = p_curr[3]; 
//             num_in_voxel++;
//         }
//     }
// }

// /***********
//  * points, Nx4
//  *
//  **************/

// void VoxelGenerator(const Eigen::MatrixXf& points, const Eigen::Vector3f& voxel_size,
//                     const Eigen::Vector3f& pc_range_max, const Eigen::Vector3f& pc_range_min,
//                     const int max_points_per_pillar, const int max_voxels, const int voxel_feature_channel,
//                     TensorFMap& voxel_info, MatrixIMap& voxel_coors, VectorIMap& voxel_num_points, int& voxel_valid_num,
//                     const bool& col_major) {
//     if (col_major) {
//         VoxelGeneratorCol(points, voxel_size, pc_range_max, pc_range_min, max_points_per_pillar, max_voxels,
//                           voxel_feature_channel, voxel_info, voxel_coors, voxel_num_points, voxel_valid_num);
//     } else {
// // 如果 points 进来的本来就是 MatrixXf (ColMajor)，这里直接转 Row 会有一次拷贝
//         // 但这是必须的，否则计算全是错的。
//         // 最优解是让上层调用者(ProcessingEigenRow)直接传 RowMatrixXf 进来
        
//         // 强制转换 view (如果上层已经传了 RowMatrixXf，这里是零拷贝)
//         // 这里的强制转换是为了匹配我上面改过的 VoxelGeneratorRow 签名
//         Eigen::Map<const RowMatrixXf> points_row(points.data(), points.rows(), points.cols());
        
//         // 警告：上面的 Map 只有在 points 本身内存连续且为 RowMajor 时才安全。
//         // 由于 VoxelGenerator 签名是 MatrixXf，这依然很危险。
//         // 建议直接修改 VoxelGenerator 的签名为泛型或 Ref。
        
//         // 【最安全的做法】：
//         // 既然你调用的入口是 VoxelGeneratorRow，建议直接修改 VoxelGeneratorRow 的签名
//         // 并在 .h 文件里对应修改。这里我们假设你已经改了 .h
        
//         // 为了编译通过，这里做一个临时的强转（会产生临时对象，因为签名不匹配）
//         // 实际上你应该去修改 .h 文件里的声明
//         RowMatrixXf points_row_copy = points; 
//         VoxelGeneratorRow(points_row_copy, voxel_size, pc_range_max, pc_range_min, max_points_per_pillar, max_voxels,
//                           voxel_feature_channel, voxel_info, voxel_coors, voxel_num_points, voxel_valid_num);
//     }
// }

// }  // namespace lidar_net
#include <cmath>
#include <iostream>

#include "log/logging.h"
#include "pillargen.h"
#include "common/deepways_time.h"

namespace lidar_net {

void VoxelGeneratorCol(const Eigen::MatrixXf& points, const Eigen::Vector3f& voxel_size,
                       const Eigen::Vector3f& pc_range_max, const Eigen::Vector3f& pc_range_min,
                       const int max_points_per_pillar, const int max_voxels, const int voxel_feature_channel,
                       TensorFMap& voxel_info, MatrixIMap& voxel_coors, VectorIMap& voxel_num_points,
                       int& voxel_valid_num) {
    Eigen::VectorXi grid_size = ((pc_range_max - pc_range_min).array() / voxel_size.array()).floor().cast<int>();
    Eigen::VectorXi coors_x = ((points.col(0).array() - pc_range_min(0)) / voxel_size(0)).floor().cast<int>();
    Eigen::VectorXi coors_y = ((points.col(1).array() - pc_range_min(1)) / voxel_size(1)).floor().cast<int>();
    Eigen::VectorXi coors_z = ((points.col(2).array() - pc_range_min(2)) / voxel_size(2)).floor().cast<int>();

    // max_voxels
    voxel_valid_num = 0;
    Eigen::MatrixXi&& coor_to_voxelidx = Eigen::MatrixXi::Constant(grid_size(0), grid_size(1), -1);

    int points_num = points.rows();
    for (int i = 0; i < points_num; ++i) {
        // TODO, mask
        if (coors_x(i) < 0 || coors_x(i) >= grid_size.x() || coors_y(i) < 0 || coors_y(i) >= grid_size.y() ||
            coors_z(i) < 0 || coors_z(i) >= grid_size.z()) {
            continue;
        }

        if (voxel_valid_num >= max_voxels) {
            ///
            return;
        }

        int& pillar_index = coor_to_voxelidx(coors_x(i), coors_y(i));

        if (pillar_index == -1) {
            // new
            pillar_index = voxel_valid_num;
            for (int c = 0; c < voxel_feature_channel; ++c) {
                voxel_info(pillar_index, voxel_num_points(pillar_index), c) = points(i, c);
            }

            voxel_coors.row(pillar_index) << 0, 0, coors_y(i), coors_x(i);
            voxel_num_points(pillar_index) += 1;
            voxel_valid_num += 1;

        } else {
            int& num = voxel_num_points(pillar_index);
            if (num < max_points_per_pillar) {
                for (int c = 0; c < voxel_feature_channel; ++c) {
                    voxel_info(pillar_index, voxel_num_points(pillar_index), c) = points(i, c);
                }
                num += 1;
            }
        }
    }
}

void VoxelGeneratorRow(const Eigen::MatrixXf& points, const Eigen::Vector3f& voxel_size,
                       const Eigen::Vector3f& pc_range_max, const Eigen::Vector3f& pc_range_min,
                       const int max_points_per_pillar, const int max_voxels, const int voxel_feature_channel,
                       TensorFMap& voxel_info, MatrixIMap& voxel_coors, VectorIMap& voxel_num_points,
                       int& voxel_valid_num) {
    Eigen::VectorXi grid_size = ((pc_range_max - pc_range_min).array() / voxel_size.array()).floor().cast<int>();
    Eigen::VectorXi coors_x = ((points.row(0).array() - pc_range_min(0)) / voxel_size(0)).floor().cast<int>();
    Eigen::VectorXi coors_y = ((points.row(1).array() - pc_range_min(1)) / voxel_size(1)).floor().cast<int>();
    Eigen::VectorXi coors_z = ((points.row(2).array() - pc_range_min(2)) / voxel_size(2)).floor().cast<int>();

    // max_voxels
    voxel_valid_num = 0;
    Eigen::MatrixXi&& coor_to_voxelidx = Eigen::MatrixXi::Constant(grid_size(0), grid_size(1), -1);

    int points_num = points.cols();
    for (int i = 0; i < points_num; ++i) {
        // TODO, mask
        if (coors_x(i) < 0 || coors_x(i) >= grid_size.x() || coors_y(i) < 0 || coors_y(i) >= grid_size.y() ||
            coors_z(i) < 0 || coors_z(i) >= grid_size.z()) {
            continue;
        }

        if (voxel_valid_num >= max_voxels) {
            ///
            return;
        }

        int& pillar_index = coor_to_voxelidx(coors_x(i), coors_y(i));

        if (pillar_index == -1) {
            // new
            pillar_index = voxel_valid_num;
            for (int c = 0; c < voxel_feature_channel; ++c) {
                voxel_info(pillar_index, voxel_num_points(pillar_index), c) = points(c, i);
            }

            voxel_coors.row(pillar_index) << 0, 0, coors_y(i), coors_x(i);
            voxel_num_points(pillar_index) += 1;
            voxel_valid_num += 1;

        } else {
            int& num = voxel_num_points(pillar_index);
            if (num < max_points_per_pillar) {
                for (int c = 0; c < voxel_feature_channel; ++c) {
                    voxel_info(pillar_index, voxel_num_points(pillar_index), c) = points(c, i);
                }
                num += 1;
            }
        }
    }
}

/***********
 * points, Nx4
 *
 **************/

void VoxelGenerator(const Eigen::MatrixXf& points, const Eigen::Vector3f& voxel_size,
                    const Eigen::Vector3f& pc_range_max, const Eigen::Vector3f& pc_range_min,
                    const int max_points_per_pillar, const int max_voxels, const int voxel_feature_channel,
                    TensorFMap& voxel_info, MatrixIMap& voxel_coors, VectorIMap& voxel_num_points, int& voxel_valid_num,
                    const bool& col_major) {
    if (col_major) {
        VoxelGeneratorCol(points, voxel_size, pc_range_max, pc_range_min, max_points_per_pillar, max_voxels,
                          voxel_feature_channel, voxel_info, voxel_coors, voxel_num_points, voxel_valid_num);
    } else {
        VoxelGeneratorRow(points, voxel_size, pc_range_max, pc_range_min, max_points_per_pillar, max_voxels,
                          voxel_feature_channel, voxel_info, voxel_coors, voxel_num_points, voxel_valid_num);
    }
}

}  // namespace lidar_net
