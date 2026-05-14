#pragma once


#include <functional>
#include <memory>
#include <utility>

#include "common/eigen_utils.h"
#include "common/json_params_get.hpp"
#include "common/json.hpp"
#include "common/thread_pool.hpp"

namespace lidar_net {

using TensorFMap = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>;
using TensorIMap = Eigen::TensorMap<Eigen::Tensor<int, 3, Eigen::RowMajor>>;
using TensorBMap = Eigen::TensorMap<Eigen::Tensor<bool, 3, Eigen::RowMajor>>;
using MatrixFMap = Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>;
using MatrixIMap = Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>>;
using MatrixBMap = Eigen::Map<Eigen::Matrix<bool, -1, -1, Eigen::RowMajor>>;
using VectorIMap = Eigen::Map<Eigen::VectorXi>;

using TensorI = Eigen::Tensor<int, 3, Eigen::RowMajor>;

class DSVTInputLayer{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DSVTInputLayer() = default;

    bool Init(const nlohmann::json& param_node);

    void doDSVTInput(const Eigen::MatrixXi& voxel_coors,
                     int* set_voxel_inds_tensor_shift_0,
                     int* set_voxel_inds_tensor_shift_1,
                     bool* set_voxel_masks_tensor_shift_0,
                     bool* set_voxel_masks_tensor_shift_1,
                     int* coors_in_win,
                     int& valid_set_num_0,
                     int& valid_set_num_1
                    );



private:
    bool getParams(const nlohmann::json& param_node);

    void windowPartition(const Eigen::MatrixXi& voxel_coors, std::vector<Eigen::VectorXi>& batch_win_inds, int* coors_in_win);

    void getSet(const std::vector<Eigen::VectorXi>& batch_win_inds,
                int* const coors_in_win,
                const int& valid_voxel_num,
                int* set_voxel_inds_tensor_shift_0,
                int* set_voxel_inds_tensor_shift_1,
                bool* set_voxel_masks_tensor_shift_0,
                bool* set_voxel_masks_tensor_shift_1,
                int& valid_set_num_0,
                int& valid_set_num_1
                );

    void getWindowCoors(const Eigen::MatrixXi& voxel_coors,
                        const Eigen::Vector3i& sparse_shape,
                        const Eigen::Vector3i& window_shape,
                        const Eigen::Vector3i& shift,
                        const bool& do_shift,
                        Eigen::VectorXi& batch_win_inds,
                        MatrixIMap& coors_in_win
                        );

    void getSetSingleShift(const Eigen::VectorXi& batch_win_inds,
                                       const MatrixIMap& coors_in_win,
                                       const int& stage_id,
                                       const int shift_id,
                                       int* set_voxel_inds_tensor,
                                       bool* set_voxel_masks,
                                       int& valid_set_num
                           );

    void getKeyMask(const MatrixIMap& set_voxel_inds, MatrixBMap& mask);

private:
    int stage_num;   // 1
    int num_shifts;  // 2
    int max_voxel_input;
    int max_win_num;
    bool flag_multi_thread = false;

    Eigen::VectorXi increasing_index_;

    // Eigen::VectorXi voxel_inds_padding_max_;

    Eigen::Vector3i sparse_shape_;  // [468, 468, 1]

    std::vector<Eigen::Vector3i> windows_shape_;  // [12, 12, 1]

    Eigen::Vector2i set_info_;      // [36, 4]

    Eigen::Vector3i hybrid_factor_;   // 2, 2, 1 : x, y, z

    std::vector<Eigen::Vector3i> shift_list_;


    ThreadPool thread_pools_ = ThreadPool(2);

};
}