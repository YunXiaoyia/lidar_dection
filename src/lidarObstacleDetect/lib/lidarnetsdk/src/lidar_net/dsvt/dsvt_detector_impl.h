#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "lidar_net_det_impl.h"
#include "lidar_net/logging.h"
#include "dsvt_params.h"

#include "dsvt_input_layer.h"

#ifdef INFERENCE_NVIDIA
#include "inference/trt/trt_infer.h"
#else
#include "infrence/mdc"
#endif

namespace lidar_net {

class DSVTDetectorImpl : public LidarNetDetectorImpl {
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

    DSVTDetectorImpl() = default;

    bool Init(const std::string_view param_node_str, std::shared_ptr<LidarNetLogger> logger, const std::string& type);

    std::vector<base::BoundingBox> Processing(const Eigen::MatrixXf& points, ProcessStatus* status, const bool& col_major = true);

    void Processing(const Eigen::MatrixXf& points, std::vector<base::BoundingBox>& objects, ProcessStatus* status, const bool& col_major = true);

    std::unique_ptr<DSVTParamV2> param_ = std::make_unique<DSVTParamV2>();

   private:
    bool JsonInit(const nlohmann::json& param_node);

    void PreProcess(const Eigen::MatrixXf& cloud, int& valid_voxel_num, int& valid_set_num_0, int& valid_set_num_1, const bool& col_major = true);

    void DoInference(int valid_voxel_num, int valid_set_num_0, int valid_set_num_1);

    void PostProcess(std::vector<base::BoundingBox>& objects);

    bool RangeROI(const Eigen::Vector3f& center, const Eigen::Vector3f& range_min, const Eigen::Vector3f& range_max);

    std::unique_ptr<infer::TrtInfer> dsvt_inference_ = std::make_unique<infer::TrtInfer>();

    DSVTInputLayer dsvt_input_layer_;

    int points_count_ = 0;

    MatrixF net_input_;
    MatrixF net_output_;

    std::vector<std::string> input_names_ = {"pillar_features",
                                             "pillar_coors",
                                             "voxel_num_points",
                                             "set_voxel_inds_tensor_shift_0",
                                             "set_voxel_inds_tensor_shift_1",
                                             "set_voxel_masks_tensor_shift_0",
                                             "set_voxel_masks_tensor_shift_1",
                                             "coors_in_win"};
    std::vector<std::vector<int>> input_shapes_;

    std::vector<std::string> output_names_ = {"pred_box"};

    int min_set_num_ = 0;
    int min_voxel_num_ = 0;

    float* pillar_features_host_buffer_;
    int* pillar_coors_host_buffer_;
    int* voxel_num_points_host_buffer_;
    int* set_voxel_inds_tensor_shift_0_host_buffer_;
    int* set_voxel_inds_tensor_shift_1_host_buffer_;
    bool* set_voxel_masks_tensor_shift_0_host_buffer_;
    bool* set_voxel_masks_tensor_shift_1_host_buffer_;
    int* coors_in_win_host_buffer_;

    float* pred_box_host_buffer_;

    std::vector<base::ObjectType> out_types_;

    Eigen::Vector3f pillar_size_;
    Eigen::Vector3i grid_size_;
    Eigen::Vector3f pc_range_min_;
    Eigen::Vector3f pc_range_max_;
    
    // /// 后处理参数
    float score_threshold;
    float iou_threshold;
    std::vector<float> iou_rectifier;
    Eigen::Matrix<float, 3, 2> roi_range_;
};

}  // namespace lidar_net
