#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "lidar_net/logging.h"
#include "lidar_net/object.h"
#include "flatformer_params.h"

#include "flattened_window_mapping.h"
#include "lidar_net_det_impl.h"

#ifdef INFERENCE_NVIDIA
#include "inference/trt/trt_infer.h"
#else
#include "infrence/mdc"
#endif

namespace lidar_net {

class FlatFormerDetectorImpl : public LidarNetDetectorImpl {
    using TensorFMap = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>;
    using TensorIMap = Eigen::TensorMap<Eigen::Tensor<int, 3, Eigen::RowMajor>>;
    using TensorBMap = Eigen::TensorMap<Eigen::Tensor<bool, 3, Eigen::RowMajor>>;
    using MatrixFMap = Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>;
    using MatrixIMap = Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>>;
    using VectorIMap = Eigen::Map<Eigen::VectorXi>;

    using Tensor = Eigen::Tensor<float, 3, Eigen::RowMajor>;
    using TensorI = Eigen::Tensor<float, 3, Eigen::RowMajor>;

    using MatrixF = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
    using MatrixI = Eigen::Matrix<int, -1, -1, Eigen::RowMajor>;

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FlatFormerDetectorImpl() = default;

    bool Init(const std::string_view param_node_str, std::shared_ptr<LidarNetLogger> logger, const std::string& type);

    std::vector<base::BoundingBox> Processing(const Eigen::MatrixXf& points, ProcessStatus* status, const bool& col_major = true);

    void Processing(const Eigen::MatrixXf& points, std::vector<base::BoundingBox>& objects, ProcessStatus* status, const bool& col_major = true);

    std::unique_ptr<FlatFormerParamV2> param_ = std::make_unique<FlatFormerParamV2>();

   private:
    bool JsonInit(const nlohmann::json& param_node);

    void PreProcess(const Eigen::MatrixXf& cloud, int& valid_voxel_num, int& valid_group_num, const bool& col_major = true);

    void DoInference(int valid_voxel_num, int valid_group_num);

    bool RangeROI(const Eigen::Vector3f& center, const Eigen::Vector3f& range_min, const Eigen::Vector3f& range_max);

    void PostProcess(std::vector<base::BoundingBox>& objects);

    std::unique_ptr<infer::TrtInfer> flatformer_inference_ = std::make_unique<infer::TrtInfer>();

    FlattenedWindowMapping flattened_window_mapping_;

    int points_count_ = 0;

    MatrixF net_input_;
    MatrixF net_output_;

    std::vector<std::string> input_names_ = {"voxels",
                                             "pillar_coords",
                                             "voxel_num_points",
                                             "flat2win",
                                             "win2flat",
                                             "mapping_x",
                                             "mapping_y",
                                             "mapping_x_shift",
                                             "mapping_y_shift"};
    std::vector<std::vector<int>> input_shapes_;

    std::vector<std::string> output_names_ = {"pred_box"};

    int min_group_num_ = 0;
    int min_voxel_num_ = 0;

    float* pillar_features_host_buffer_;
    int* pillar_coors_host_buffer_;
    int* voxel_num_points_host_buffer_;
    int* flat2win_host_buffer_;
    int* win2flat_host_buffer_;
    int* mapping_x_host_buffer_;
    int* mapping_y_host_buffer_;
    int* mapping_x_shift_host_buffer_;
    int* mapping_y_shift_host_buffer_;

    float* pred_box_host_buffer_;

    std::vector<base::ObjectType> out_types_;

    Eigen::Vector3f pillar_size_;
    Eigen::Vector3i grid_size_;
    Eigen::Vector3f pc_range_min_;
    Eigen::Vector3f pc_range_max_;

    float score_threshold;
    float iou_threshold;
    std::vector<float> iou_rectifier;
    Eigen::Matrix<float, 3, 2> roi_range_;
};

}  // namespace lidar_net
