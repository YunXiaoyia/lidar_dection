#pragma once

#include <exception>
#include <unsupported/Eigen/CXX11/Tensor>

#include "log/logger.h"
#include "common/json_params_get.hpp"
#include "common/json.hpp"

namespace lidar_net {

struct DSVTParamV2 {
    DSVTParamV2() = default;

    bool Init(const nlohmann::json& param_node) noexcept {
        try {
            GET_VALUE(param_node, pointcloud_range);

            GET_VALUE(param_node, voxel_size);

            GET_VALUE(param_node, max_input_points);
            GET_VALUE(param_node, input_points_channel);
            GET_VALUE(param_node, max_voxel_input);
            GET_VALUE(param_node, max_point_per_voxel);
            GET_VALUE(param_node, max_set_num);

            GET_VALUE(param_node, output_rows);
            GET_VALUE(param_node, output_cols);
           
            GET_VALUE(param_node, num_classes);
           
            GET_VALUE(param_node, min_set_num);
            GET_VALUE(param_node, min_voxel_num);

            return true;
        } catch (const std::exception& e) {
            PLOG_FATAL << "[Detector] DSVT Param JSON config failed with: " << e.what();
        }
        return false;
    }

    std::vector<float> pointcloud_range;
    std::vector<float> voxel_size;

    int max_input_points;
    int input_points_channel;
    int max_voxel_input;
    int max_point_per_voxel;
    int max_set_num;

    int output_rows;
    int output_cols;

    int num_classes;

    int min_set_num;
    int min_voxel_num;
};

}  // namespace lidar_net
