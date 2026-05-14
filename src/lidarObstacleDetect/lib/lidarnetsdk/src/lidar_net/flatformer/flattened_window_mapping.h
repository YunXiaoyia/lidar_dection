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

class FlattenedWindowMapping{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FlattenedWindowMapping() = default;

    bool Init(const nlohmann::json& param_node);

    void doMapping(const Eigen::MatrixXi& voxel_coors,
                     int* flat2win,
                     int* win2flat,
                     int* mapping_x,
                     int* mapping_y,
                     int* mapping_x_shift,
                     int* mapping_y_shift,
                     int& valid_group_num
                    );

    void getWindFlatMapping(const int& voxel_num,
                            const int& padded_voxel_num,
                            MatrixIMap& flat2win_map,
                            VectorIMap& win2flat_map);

    void getWindowCoorsShift(const Eigen::MatrixXi& coords,
                        const bool& shifted,
                        VectorIMap& mapping_x,
                        VectorIMap& mapping_y);


private:
    bool getParams(const nlohmann::json& param_node);


private:
    int num_shifts;  // 2
    int max_voxel_input;
    int max_group_num;
    int group_size;
    bool flag_multi_thread = false;


    Eigen::Vector3i sparse_shape_;
    Eigen::Vector3i windows_shape_;
    Eigen::Vector3i windows_num_;
    Eigen::Vector3i coor_shift_;

    Eigen::Vector3i windows_shape_2_;
    Eigen::Vector3i windows_num_2_;


    Eigen::VectorXi increasing_index_;

    // Eigen::VectorXi voxel_inds_padding_max_;


    Eigen::Vector2i set_info_;      // [36, 4]

    Eigen::Vector3i hybrid_factor_;   // 2, 2, 1 : x, y, z

    std::vector<Eigen::Vector3i> shift_list_;


    ThreadPool thread_pools_ = ThreadPool(2);

};
}