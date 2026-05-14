#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "common/eigen_utils.h"
#include "log/logging.h"
#include "flatformer_detector_impl.h"
#include "utils/config_convert.h"
#include "utils/nms_utils.h"
#include "utils/pillargen.h"
#include "inference/trt/trt_infer.h"
#include "log/logger.h"
#include "common/deepways_time.h"
#include "common/json_params_get.hpp"

namespace lidar_net {


bool FlatFormerDetectorImpl::JsonInit(const nlohmann::json& param_node) {
    const auto& flatformer_param_node = param_node["flatformer"];
    if (!flatformer_param_node.is_object()) {
        PLOG_FATAL << "[Detector] flatformer param is not a map!";
        return false;
    }

    // Init params
    const auto& params_node = flatformer_param_node["params"];
    if (!param_->Init(params_node)) {
        PLOG_FATAL << "[Detector] init FlatFormerDetectorImpl Params Unsuccessfully";
        return false;
    }

    // Init Output Types
    const auto out_types_node = flatformer_param_node["out_types"];
    if (!out_types_node.is_array()) {
        PLOG_FATAL << "[Detector] toml config out_types is not properly set.";
        return false;
    } else {
        auto out_types_arr = out_types_node.get<std::vector<std::string>>();
        if (out_types_arr.size() != static_cast<size_t>(param_->num_classes)) {
            PLOG_FATAL << "[Detector] toml config out_types size is not equal to num_classes.";
            return false;
        }
        for (size_t i = 0; i < out_types_arr.size(); ++i) {
            out_types_.push_back(base::ObjectTypeDict.at(out_types_arr[i]));
        }
    }

    // Init FlatFormer Mapping
    const auto flattened_window_mapping_node = flatformer_param_node["flattened_window_mapping"];
    if(!flattened_window_mapping_.Init(flattened_window_mapping_node)){
        PLOG_FATAL << "[Detector] init FlattenedWindowMapping Params Unsuccessfully";
        return false;
    }

    // 后处理参数
    const auto postprocess_node = flatformer_param_node["network"]["postprocess"];
    GET_VALUE(postprocess_node, score_threshold);
    GET_VALUE(postprocess_node, iou_threshold);
    GET_VALUE(postprocess_node, iou_rectifier);

    std::vector<float> roi_range;
    GET_VALUE(postprocess_node, roi_range);
    roi_range_.col(0) << roi_range[0], roi_range[1], roi_range[2];
    roi_range_.col(1) << roi_range[3], roi_range[4], roi_range[5];


    return true;
}

bool FlatFormerDetectorImpl::Init(const std::string_view param_node_str, std::shared_ptr<LidarNetLogger> logger,
                               const std::string& config_type) {
    auto param_node_json = config_convert(param_node_str, config_type);

    // pLogger = logger;

    bool ret = true;
    std::stringstream os;
    const nlohmann::json config = nlohmann::json::parse(param_node_json);
    ret = this->JsonInit(config);
    os << config["flatformer"]["network"];

    if (!ret) return false;

    if (!flatformer_inference_->Init(os.str(), "json")) {
        PLOG_FATAL << "[Detector] init FlatFormerDetectorImpl Inference Unsuccessfully";
        return false;
    }

    // input shape
    {
        input_shapes_.push_back(std::vector<int>({param_->max_voxel_input, param_->max_point_per_voxel, param_->input_points_channel}));         // 15000, 20, 4
        input_shapes_.push_back(std::vector<int>({param_->max_voxel_input, 4}));
        input_shapes_.push_back(std::vector<int>({param_->max_voxel_input}));
        input_shapes_.push_back(std::vector<int>({param_->max_group_num, param_->group_size})); // flat2win
        input_shapes_.push_back(std::vector<int>({param_->max_voxel_input}));                   // win2flat
        input_shapes_.push_back(std::vector<int>({param_->max_voxel_input}));                   // mapping_x
        input_shapes_.push_back(std::vector<int>({param_->max_voxel_input}));                   // mapping_y
        input_shapes_.push_back(std::vector<int>({param_->max_voxel_input}));                   // mapping_x_shift
        input_shapes_.push_back(std::vector<int>({param_->max_voxel_input}));                   // mapping_y_shift
    }

    min_group_num_ = param_->min_group_num;
    min_voxel_num_ = param_->min_voxel_num;

    // get host_buffer
    pillar_features_host_buffer_ = static_cast<float*>(flatformer_inference_->GetInput("voxels"));
    pillar_coors_host_buffer_ = static_cast<int*>(flatformer_inference_->GetInput("pillar_coords"));
    voxel_num_points_host_buffer_ = static_cast<int*>(flatformer_inference_->GetInput("voxel_num_points"));
    flat2win_host_buffer_ = static_cast<int*>(flatformer_inference_->GetInput("flat2win"));
    win2flat_host_buffer_ = static_cast<int*>(flatformer_inference_->GetInput("win2flat"));
    mapping_x_host_buffer_ = static_cast<int*>(flatformer_inference_->GetInput("mapping_x"));
    mapping_y_host_buffer_ = static_cast<int*>(flatformer_inference_->GetInput("mapping_y"));
    mapping_x_shift_host_buffer_ = static_cast<int*>(flatformer_inference_->GetInput("mapping_x_shift"));
    mapping_y_shift_host_buffer_ = static_cast<int*>(flatformer_inference_->GetInput("mapping_y_shift"));

    pred_box_host_buffer_ = static_cast<float*>(flatformer_inference_->GetOutput("pred_box"));

    // Init Local Params
    pillar_size_ = Eigen::Vector3f{param_->voxel_size[0], param_->voxel_size[1], param_->voxel_size[2]};
    pc_range_min_ = Eigen::Vector3f{param_->pointcloud_range[0], param_->pointcloud_range[1], param_->pointcloud_range[2]};
    pc_range_max_ = Eigen::Vector3f{param_->pointcloud_range[3], param_->pointcloud_range[4], param_->pointcloud_range[5]};
    grid_size_ = ((pc_range_max_ - pc_range_min_).array() / pillar_size_.array()).floor().cast<int>();
    return true;
}

std::vector<base::BoundingBox> FlatFormerDetectorImpl::Processing(const Eigen::MatrixXf& cloud, ProcessStatus* status, const bool& col_major) {
    std::vector<base::BoundingBox> objects;

    this->Processing(cloud, objects, status, col_major);

    return objects;
}

void FlatFormerDetectorImpl::Processing(const Eigen::MatrixXf& cloud, std::vector<base::BoundingBox>& objects, ProcessStatus* status, const bool& col_major) {
    PERF_BLOCK_START();
    int valid_voxel_num = 0;
    int valid_group_num = 0;

    this->PreProcess(cloud, valid_voxel_num, valid_group_num, col_major);
    if (valid_voxel_num < this->min_voxel_num_  || valid_group_num < this->min_group_num_) {
        std::stringstream error;
        error <<  "[Detector] preprocess invalid input:\n" <<
                    "valid_voxel_num:" << valid_voxel_num << ", min_voxel_num:" << this->min_voxel_num_ << "\n" <<
                    "valid_group_num_0:" << valid_group_num << ", min_group_num:" << this->min_group_num_;

        if (status) {
            status->status_code = StatusCode::InputError;
            status->log = error.str();
        }
        PLOG_WARN << error.str();
        return;
    }
    PERF_BLOCK_END("FlatFormer PreProcess");

    this->DoInference(valid_voxel_num, valid_group_num);
    PERF_BLOCK_END("FlatFormer DoInference");

    this->PostProcess(objects);
    PERF_BLOCK_END("FlatFormer PostProcess");

}

void FlatFormerDetectorImpl::PreProcess(const Eigen::MatrixXf& cloud, int& valid_voxel_num, int& valid_group_num, const bool& col_major) {

    memset(pillar_features_host_buffer_, 0, param_->max_voxel_input * param_->max_point_per_voxel * param_->input_points_channel * sizeof(float));
    memset(pillar_coors_host_buffer_,0, param_->max_voxel_input * sizeof(int));
    memset(voxel_num_points_host_buffer_,0, param_->max_voxel_input *sizeof(int));
    memset(flat2win_host_buffer_, 0, param_->max_group_num * param_->group_size * sizeof(int));
    memset(win2flat_host_buffer_,0, param_->max_voxel_input * sizeof(int));
    memset(mapping_x_host_buffer_,0, param_->max_voxel_input *sizeof(int));
    memset(mapping_y_host_buffer_,0, param_->max_voxel_input *sizeof(int));
    memset(mapping_x_shift_host_buffer_,0, param_->max_voxel_input *sizeof(int));
    memset(mapping_y_shift_host_buffer_,0, param_->max_voxel_input *sizeof(int));

    // generate pillar
    TensorFMap pillar_features(pillar_features_host_buffer_, param_->max_voxel_input, param_->max_point_per_voxel, param_->input_points_channel);
    MatrixIMap pillar_coors(pillar_coors_host_buffer_, param_->max_voxel_input, 4);
    VectorIMap voxel_num_points(voxel_num_points_host_buffer_, param_->max_voxel_input);
    VoxelGenerator(cloud, pillar_size_, pc_range_max_, pc_range_min_,
                param_->max_point_per_voxel, param_->max_voxel_input,
                param_->input_points_channel, pillar_features, pillar_coors, voxel_num_points, valid_voxel_num, col_major);

    const Eigen::MatrixXi& tmp_coors = pillar_coors.topRows(valid_voxel_num);
    // PERF_BLOCK_START();

    flattened_window_mapping_.doMapping(tmp_coors,
                             flat2win_host_buffer_,
                             win2flat_host_buffer_,
                             mapping_x_host_buffer_,
                             mapping_y_host_buffer_,
                              mapping_x_shift_host_buffer_,
                              mapping_y_shift_host_buffer_,
                              valid_group_num);

    // PERF_BLOCK_END("PreProcess doMapping");
}

void FlatFormerDetectorImpl::DoInference(int valid_voxel_num, int valid_group_num){
    input_shapes_[0][0] = valid_voxel_num;
    input_shapes_[1][0] = valid_voxel_num;
    input_shapes_[2][0] = valid_voxel_num;

    input_shapes_[3][0] = valid_group_num;
    input_shapes_[4][0] = valid_voxel_num;

    input_shapes_[5][0] = valid_voxel_num;
    input_shapes_[6][0] = valid_voxel_num;
    input_shapes_[7][0] = valid_voxel_num;
    input_shapes_[8][0] = valid_voxel_num;

    flatformer_inference_->InferDynamicInput(input_names_, input_shapes_);
}

void FlatFormerDetectorImpl::PostProcess(std::vector<base::BoundingBox>& objects) {
    objects.clear();

    // MatrixMap
    MatrixFMap net_output(pred_box_host_buffer_, param_->output_rows, param_->output_cols);
    std::vector<base::BoundingBox> bbox_vec;
    for(int i = 0; i < net_output.rows(); i++) {
        const Eigen::VectorXf outobj = net_output.row(i);

        // [x, y, z, dx, dy, dz, rot_sin, rot_cos, score, class_id, iou]
        // outobj(10) > 0.01 为了防止iou出现负值
        if(outobj(8) >= score_threshold && outobj(10) > 0.01){
            base::BoundingBox bbox;
            bbox.center.x() = outobj(0);
            bbox.center.y() = outobj(1);
            bbox.center.z() = outobj(2);
            bbox.size.x() = outobj(3);
            bbox.size.y() = outobj(4);
            bbox.size.z() = outobj(5);

            bbox.theta = std::atan2(outobj(6), outobj(7));
            bbox.iou = outobj(10);
            bbox.confidence = outobj(8);
            float iou_ref = iou_rectifier[static_cast<int>(outobj(9))];
            bbox.confidence_iou = std::pow(outobj(8), (1.0-iou_ref)) * std::pow(outobj(10), iou_ref);   //  outobj(8);
            bbox.type = out_types_[static_cast<int>(outobj(9))];

            /// 计算4个角点
            Eigen::Rotation2D<float> R2D(bbox.theta);
            bbox.corners2d.col(0) = R2D * Eigen::Vector2f(bbox.size(0), bbox.size(1)) * 0.5f + bbox.center.head(2);
            bbox.corners2d.col(1) = R2D * Eigen::Vector2f(bbox.size(0), -bbox.size(1)) * 0.5f + bbox.center.head(2);
            bbox.corners2d.col(2) = R2D * Eigen::Vector2f(-bbox.size(0), -bbox.size(1)) * 0.5f + bbox.center.head(2);
            bbox.corners2d.col(3) = R2D * Eigen::Vector2f(-bbox.size(0), bbox.size(1)) * 0.5f + bbox.center.head(2);

            bbox_vec.emplace_back(bbox);
        }
    }
    nms(bbox_vec, iou_threshold);

    const size_t sz_bbox = bbox_vec.size();
    objects.reserve(sz_bbox);

    for (size_t idx = 0; idx < sz_bbox; ++idx) {
        base::BoundingBox object;

        auto& bbox = bbox_vec[idx];
        /// 按距离从小到大，重新排列角点
        int nearest_id = 0;
        bbox.corners2d.colwise().squaredNorm().minCoeff(&nearest_id);
        if (nearest_id != 0) {
            const Eigen::Matrix<float, 2, 4> temp = bbox.corners2d;
            const int sz = bbox.corners2d.cols();
            for (int i = 0; i < sz; ++i) {
                bbox.corners2d.col(i) = temp.col((nearest_id + i) % sz);
            }
        }
        // 中心是否在距离范围内
        if (!RangeROI(bbox.center, roi_range_.col(0), roi_range_.col(1))) {
            continue;
        }

        objects.push_back(bbox);
    }

}

bool FlatFormerDetectorImpl::RangeROI(const Eigen::Vector3f& center, const Eigen::Vector3f& range_min, const Eigen::Vector3f& range_max){
    bool flag = (center.array() > range_min.array()).all();
    flag &= (center.array() < range_max.array()).all();
    return flag;
}


}  // namespace lidar_net
