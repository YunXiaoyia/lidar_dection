#include "simple_track/tracklet.h"
#include <memory>
#include <iostream>

// 临时定义 TLOG_WARN 宏
#ifndef TLOG_WARN
#define TLOG_WARN std::cout << "[WARN] "
#endif

/* Headers for shape filter */
#include "../../lidarnetsdk/include/lidar_net/object.h"
#include "shape_filter/moving_average/moving_average_filter.h"

/* Headers for motion filter */
#include "motion_filter/l_shape/lshape_motion_filter.h"
#include "motion_filter/multi_anchor/multi_anchor_motion_filter.h"


#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

// Init static parameters
int Tracklet::MAX_OBJECT_BUFFER_NUM = 20;
int Tracklet::MAX_TRACK_BUFFER_NUM = 20;
double Tracklet::VALID_LOST_DURATION = 0.4;
int Tracklet::MAX_CONTINUOUS_LOST_NUM = 5;
float Tracklet::FOV_BOUND_DIS = 3.0;
int Tracklet::MIN_CONTINUOUS_DET_NUM = 2;

std::string Tracklet::SHAPE_FILTER_TYPE = "MovingAverageShapeFilter";
std::string Tracklet::MOTION_FILTER_TYPE = "MultiAnchorMotionFilter";

bool Tracklet::InitStaticParams(const toml::node_view<const toml::node>& param_node) {

    // Init motion filter
    std::string motion_type = "Multi_anchor";
    // auto motion_type = param_node.at_path("motion_filter").value<std::string>();
    if (!param_node["motion_filter"].is_value()) {
        TLOG_WARN << "[Tracklet] Dont't find Motion filter definition! please check param: motion_filter."
                     << " use default: Multi_anchor.";
    } else {
        motion_type = param_node["motion_filter"].value<std::string>().value();
    }


    if (motion_type == "multi_anchor" && param_node["motion"]["multi_anchor"].is_table()) {
        MultiAnchorMotionFilter::InitStaticParams(param_node["motion"]["multi_anchor"]);
        MOTION_FILTER_TYPE = "MultiAnchorMotionFilter";
    } else if (motion_type == "l_shape" && param_node["motion"]["l_shape"].is_table()) {
        LShapeMotionFilter::InitStaticParams(param_node["motion"]["l_shape"]);
        MOTION_FILTER_TYPE = "LShapeMotionFilter";
    }

    // Init shape filter
    std::string shape_type = "moving_average";
    if (!param_node["shape_filter"].is_value()) {
        TLOG_WARN << "[Tracklet] Dont't find Shape filter definition! please check param: shape_filter."
                     << " use default: moving_average.";
    } else {
        shape_type = param_node["shape_filter"].value<std::string>().value();
    }

    if (shape_type == "moving_average" && param_node["shape"]["moving_average"].is_table()) {
        MovingAverageShapeFilter::InitStaticParams(param_node["shape"]["moving_average"]);
        SHAPE_FILTER_TYPE = "MovingAverageShapeFilter";
    }

    // Init tracklet self params
    if (!param_node["tracklet"].is_table()) { return true; }
    auto trk_params = param_node["tracklet"];
    auto valid_lost_duraion = trk_params.at_path("valid_lost_duraion").value<float>();
    if (!valid_lost_duraion.has_value()) {
        TLOG_WARN << "[Tracklet] Dont't find VALID_LOST_DURATION definition! please check param: valid_lost_duraion."
                     << " use default:" << VALID_LOST_DURATION;
    } else {
        VALID_LOST_DURATION = valid_lost_duraion.value();
    }

    auto max_continuous_lost_num = trk_params.at_path("max_continuous_lost_num").value<int>();
    if (!max_continuous_lost_num.has_value()) {
        TLOG_WARN << "[Tracklet] Dont't find MAX_CONTINUOUS_LOST_NUM definition! please check param: max_continuous_lost_num."
                     << " use default:" << MAX_CONTINUOUS_LOST_NUM;
    } else {
        MAX_CONTINUOUS_LOST_NUM = max_continuous_lost_num.value();
    }

    auto min_continuous_det_num = trk_params.at_path("min_continuous_det_num").value<int>();
    if (!min_continuous_det_num.has_value()) {
        TLOG_WARN << "[Tracklet] Dont't find MIN_CONTINUOUS_DET_NUM definition! please check param: min_continuous_det_num."
                     << " use default:" << MIN_CONTINUOUS_DET_NUM;
    } else {
        MIN_CONTINUOUS_DET_NUM = min_continuous_det_num.value();
    }


    auto fov_bound_dis = trk_params.at_path("fov_bound_dis").value<float>();
    if (!fov_bound_dis.has_value()) {
        TLOG_WARN << "[Tracklet] Dont't find FOV_BOUND_DIS definition! please check param: fov_bound_dis."
                     << " use default:" << FOV_BOUND_DIS;
    } else {
        FOV_BOUND_DIS = fov_bound_dis.value();
    }

    return true;
}


bool Tracklet::inFovBound(const common::ObjectPtr& object) {

    float bound_x = fabs(object->bbox.center.y() * fov_tan_theta_);

    float tail_x = object->bbox.center.x() - object->bbox.size.x() * 0.5 - x_correction_offset_;

    return tail_x < (bound_x + FOV_BOUND_DIS);
}

Tracklet::Tracklet() {

    // Motion filter
    motion_filter_ = BaseTrackFilterRegisterer::GetInstanceByName(MOTION_FILTER_TYPE);

    // Shape filter
    shape_filter_ = BaseTrackFilterRegisterer::GetInstanceByName(SHAPE_FILTER_TYPE);
}

void Tracklet::Init(double timestamp, const common::ObjectPtr& object, const TrackletInitParams& init_params) {
    fov_tan_theta_ = std::tan(3.1415926 * 0.5 - 0.5 * init_params.fov_theta / 180.0 * 3.1415926);
    x_correction_offset_ = init_params.x_direction_offset;

    // Init tracked object
    TrackedObjectPtr new_tracked_object = std::make_shared<TrackedObject>();
    new_tracked_object->in_fov_bound = inFovBound(object);
    
    new_tracked_object->AttachObject(object);

    new_tracked_object->output_center = new_tracked_object->center;
    new_tracked_object->output_size = new_tracked_object->size;
    // 使用 theta 计算方向向量
    new_tracked_object->output_direction = Eigen::Vector3f(
        cos(new_tracked_object->object_ptr->bbox.theta), 
        sin(new_tracked_object->object_ptr->bbox.theta), 
        0.0f
    );

    TrackFilterInitOptions init_opt(new_tracked_object);
    motion_filter_->Init(init_opt);

    last_tracked_time_ = timestamp;

    track_age_ = 1;

    consecutive_invisible_count_ = 0;

    consecutive_visible_count_ = 0;

    // Cache object info
    PushTrackedObjectToTrack(new_tracked_object);
    PushDetectedObjectToDetect(object);
}

void Tracklet::TransformToCurrentFrame(const Eigen::Isometry3f& tf) {

    if (history_tracked_objects_.empty()) { return; }

    // TLOG_INFO << "track id:" << track_id_ << " age:" << track_age_ << " size:" << history_tracked_objects_.size();

    for (auto& track : history_tracked_objects_) {

        // Output info
        track->output_center = tf * track->output_center;
        track->output_velocity = tf.rotation() * track->output_velocity;
        track->output_corners[0] = (tf * track->output_corners[0]).eval();
        track->output_corners[1] = (tf * track->output_corners[1]).eval();
        track->output_corners[2] = (tf * track->output_corners[2]).eval();
        track->output_corners[3] = (tf * track->output_corners[3]).eval();
        track->output_direction = tf.rotation() * track->output_direction;
        track->output_theta = std::atan2(track->output_direction.y(), track->output_direction.x());

        track->output_selected_track_point = tf * track->output_selected_track_point;

        // Belief info
        track->anchor_point = tf * track->anchor_point;  // 速度估算中使用
        track->selected_track_point = tf * track->selected_track_point;  // 速度估算中使用

        // Points
        track->object_ptr->points->getMatrixXfMap(3, 8, 0) =
                tf * track->object_ptr->points->getMatrixXfMap(3, 8, 0).colwise().homogeneous();
    }

    // std::cout << "back transted pos: " << history_tracked_objects_.back()->output_selected_track_point[0] << " "
    //                               << history_tracked_objects_.back()->output_selected_track_point[1] << " "
    //                               << history_tracked_objects_.back()->output_selected_track_point[2] << " "  << std::endl;

    motion_filter_->TransState(tf);
}

TrackedObjectConstPtr Tracklet::GetLatestTrackedObject() {

    if (history_tracked_objects_.empty()) {
        return nullptr;
    } else {
        return history_tracked_objects_.back();
    }
}

void Tracklet::updateObjectType(common::ObjectType& object_type){
    if(history_detected_objects_.size() > 5){
        Eigen::VectorXi type_counter = Eigen::VectorXi::Zero(static_cast<int>(common::ObjectType::TYPE_NUM));
        for(auto& obj : history_detected_objects_){
            int type_idx = static_cast<int>(obj->type);
            ++type_counter[type_idx];
        }

        int class_idx = 0;
        type_counter.maxCoeff(&class_idx);
        object_type = static_cast<common::ObjectType>(class_idx);
    }
    // else{
    //     object_type = history_tracked_objects_.back()->output_type;
    // }
}

TrackedObjectConstPtr Tracklet::GetOutTrackedObject() {
    auto out_obj = std::make_shared<TrackedObject>();

    if (history_tracked_objects_.empty()) {
        return nullptr;
    } else if (consecutive_invisible_count_ > 0) {

        double dt = last_untracked_time_ - last_tracked_time_;

        // If time duration lest than 5ms, not predict
        if (dt < 0.005) { return history_tracked_objects_.back(); }

        const auto& last_obj = history_tracked_objects_.back();

        float delta_x = last_obj->output_velocity.x() * dt;
        float delta_y = last_obj->output_velocity.y() * dt;
        Eigen::Vector3f delta(delta_x, delta_y, 0.0f);

        // TLOG_INFO << "track id:" << track_id_ << " age:" << track_age_ << " " << delta[0] << " " << delta[1] << " dt:" << dt;

        out_obj->output_type = last_obj->output_type;
        out_obj->output_velocity = last_obj->output_velocity;

        // Copy boundingbox info
        out_obj->output_corners[0] = last_obj->output_corners[0] + delta;
        out_obj->output_corners[1] = last_obj->output_corners[1] + delta;
        out_obj->output_corners[2] = last_obj->output_corners[2] + delta;
        out_obj->output_corners[3] = last_obj->output_corners[3] + delta;
        out_obj->output_size = last_obj->output_size;
        out_obj->output_center = last_obj->output_center + delta;
        out_obj->output_theta = last_obj->output_theta;
        out_obj->output_state_covariance = last_obj->output_state_covariance;

        out_obj->output_direction = last_obj->output_direction;
        out_obj->timestamp = last_obj->timestamp;

        return out_obj;

    } else {
        return history_tracked_objects_.back();
    }
}

common::ObjectPtr Tracklet::GetLatestDetectedObject() {
    if (history_detected_objects_.empty()) {
        return nullptr;
    } else {
        return history_detected_objects_.back();
    }

}

void Tracklet::PushTrackedObjectToTrack(const TrackedObjectPtr& new_tracked) {
    history_tracked_objects_.push_back(new_tracked);

    if (history_tracked_objects_.size() > uint(MAX_TRACK_BUFFER_NUM)) {
        history_tracked_objects_.pop_front();
    }
}

void Tracklet::PushDetectedObjectToDetect(const common::ObjectPtr& new_detected) {

    history_detected_objects_.push_back(new_detected);

    if (history_detected_objects_.size() > uint(MAX_OBJECT_BUFFER_NUM)) {
        history_detected_objects_.pop_front();
    }
}

void Tracklet::updateWithObject(const common::ObjectPtr& detected_object) {
    // Extract latest tracked object
    TrackedObjectConstPtr latest_object = this->GetLatestTrackedObject();

    // Convert to tracked object
    TrackedObjectPtr new_object = std::make_shared<TrackedObject>();

    new_object->track_id = track_id_;
    new_object->in_fov_bound = inFovBound(detected_object);
    new_object->AttachObject(detected_object);

    // 1. State filter and store belief in new_object
    if (shape_filter_) {
        shape_filter_->UpdateWithObject(latest_object, new_object);
    }
    
    if (motion_filter_) {
        motion_filter_->UpdateWithObject(latest_object, new_object);
    }

    // type filter
    updateObjectType(new_object->output_type);

    // 2. Push new_obect to track_data and cach detect object
    PushTrackedObjectToTrack(new_object);
    PushDetectedObjectToDetect(detected_object);

    // 3. Update info for tracked
    updateTrackInfo(new_object);
}

void Tracklet::updateTrackInfo(TrackedObjectPtr& tracked_object) {
    track_age_++;

    consecutive_visible_count_++;

    last_tracked_time_ = tracked_object->timestamp;

    consecutive_invisible_count_ = 0;

    // //  Upgrade status
    // if (consecutive_visible_count_ > 3) {
    //     status_ = TrackletStatus::CONFIRMED;
    // }
}

void Tracklet::updateWithoutObject(const double& timestamp) {

    track_age_++;

    consecutive_invisible_count_++;

    consecutive_visible_count_ = 0;

    last_untracked_time_ = timestamp;

    double delta_time = timestamp - last_tracked_time_;
    if (delta_time > VALID_LOST_DURATION) {
        status_ = TrackletStatus::DELETED;
    }
    
    if (!history_tracked_objects_.empty()) { 
        TrackedObjectPtr latest_object = history_tracked_objects_.back();
        if (latest_object) latest_object->timestamp = timestamp; 
    }
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
