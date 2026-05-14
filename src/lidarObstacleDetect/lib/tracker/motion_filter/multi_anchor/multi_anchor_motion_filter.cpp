#include "motion_filter/multi_anchor/multi_anchor_motion_filter.h"
#include <cmath>
#include <string>
#include "motion_filter/kalman_model/extended_kalman_cv_model.hpp"
#include "motion_filter/kalman_model/extended_kalman_ca_model.hpp"
#include "motion_filter/kalman_model/extended_kalman_ca_model_without_velocity.hpp"
#include "motion_filter/kalman_model/normal_kalman_cv_model.hpp"
#include "lib/utils/kalman/SimpleKalmanFilter.hpp"
#include "common/lidar_perception_log.h"


#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

MultiAnchorParams MultiAnchorMotionFilter::params_;

bool MultiAnchorMotionFilter::InitStaticParams(const toml::node_view<const toml::node>& param_node) {

    auto kinematics_model = param_node.at_path("kinematics_model").value<std::string>();
    if (!kinematics_model.has_value()) {
        TLOG_WARN << "[MultiAnchorMotionFilter] Dont't find kinematics_model definition! please check param: kinematics_model."
                     << " use default:" << params_.filter_type;
    } else {
        params_.filter_type = kinematics_model.value();
    }

    auto velo_head_change = param_node.at_path("velo_head_change").value<float>();
    if (!velo_head_change.has_value()) {
        TLOG_WARN << "[MultiAnchorMotionFilter] Dont't find velo_head_change definition! please check param: velo_head_change."
                     << " use default:" << params_.velocity_angle_change_threshold;
    } else {
        params_.velocity_angle_change_threshold = velo_head_change.value() / 180.0 * M_PI;
    }

    auto velo_norm_change = param_node.at_path("velo_norm_change").value<float>();
    if (!velo_norm_change.has_value()) {
        TLOG_WARN << "[MultiAnchorMotionFilter] Dont't find velo_norm_change definition! please check param: velo_norm_change."
                     << " use default:" << params_.velocity_length_change_threshold;
    } else {
        params_.velocity_length_change_threshold = velo_norm_change.value();
    }

    auto acc_head_change = param_node.at_path("acc_head_change").value<float>();
    if (!acc_head_change.has_value()) {
        TLOG_WARN << "[MultiAnchorMotionFilter] Dont't find acc_head_change definition! please check param: acc_head_change."
                     << " use default:" << params_.acceleration_angle_change_threshold;
    } else {
        params_.acceleration_angle_change_threshold = acc_head_change.value() / 180.0 * M_PI;
    }

    auto eval_window_size = param_node.at_path("eval_window_size").value<int>();
    if (!eval_window_size.has_value()) {
        TLOG_WARN << "[MultiAnchorMotionFilter] Dont't find eval_window_size definition! please check param: eval_window_size."
                     << " use default:" << params_.sensor_eval_window_size;
    } else {
        params_.sensor_eval_window_size = eval_window_size.value();
    }

    return true;
}

bool MultiAnchorMotionFilter::Init(const TrackFilterInitOptions& options)
{
    motion_meas_.reset(new MultiAnchorMotionMeasurement());
    
    // Try to get kalman filter
    auto kalman_instance = IKalmanFilterRegisterer::GetInstanceByName(params_.filter_type);
    
    if (!kalman_instance) {
        kalman_filter_.reset(new Kalman::SimpleCVKalmanFilter());
    } else {
        kalman_filter_.reset(kalman_instance.get());
    }

    options.tracked_object->selected_measured_velocity = Eigen::Vector3f::Zero();
    options.tracked_object->output_selected_track_point = options.tracked_object->selected_track_point;

    // Update velocity
    options.tracked_object->output_velocity = Eigen::Vector3f::Zero();

    // InitFilter(options.tracked_object);

    // last_predict_timestamp_ = options.tracked_object->timestamp;
    // [Fix] 显式初始化成员变量，防止垃圾值导致的逻辑错误
    fused_track_point_.setZero();
    fused_velocity_.setZero();
    fused_acceleration_.setZero();
    filter_init_ = false; 

    InitFilter(options.tracked_object);

    last_predict_timestamp_ = options.tracked_object->timestamp;
    return true;
}

void MultiAnchorMotionFilter::TransState(const Eigen::Isometry3f& tf) {
    fused_track_point_ = tf * fused_track_point_;

    fused_velocity_ = tf.rotation() * fused_velocity_;
    fused_acceleration_ = tf.rotation() * fused_acceleration_;
    if (filter_init_) {
        kalman_filter_->StateChange(tf);
    }
}

void MultiAnchorMotionFilter::StateChange(const TrackedObjectConstPtr& track_data,
                                          TrackedObjectPtr& measurement) {
    if (measurement->isTailMiddlePoint()) {
        if (track_data->in_fov_bound && !measurement->in_fov_bound) {
            float offset_x = -track_data->size.x();
            kalman_filter_->StateChange(offset_x, 0.0);
        }

        if (!track_data->in_fov_bound && measurement->in_fov_bound) {
            float offset_x = track_data->size.x();
            kalman_filter_->StateChange(offset_x, 0.0);
        }
    }
}

bool MultiAnchorMotionFilter::InitFilter(const TrackedObjectConstPtr& sensor_object)
{

    if (!fused_track_point_.isZero() && fused_velocity_.isZero()) {
        double time_diff = sensor_object->timestamp - last_predict_timestamp_;
        time_diff = time_diff > 0.001 ? time_diff : 0.001;

        if(sensor_object->isTailMiddlePoint()){
            if(in_fov_bound_ && !sensor_object->in_fov_bound){
                float offset_x = sensor_object->size.x();
                fused_track_point_.x() -= offset_x;
            }

            if(!in_fov_bound_ && sensor_object->in_fov_bound){
                float offset_x = sensor_object->size.x();
                fused_track_point_.x() += offset_x;
            }
        }

        fused_velocity_ = (sensor_object->velo_anchor_point - fused_track_point_) / time_diff;
        fused_track_point_ = sensor_object->velo_anchor_point;
        fused_acceleration_ = Eigen::Vector3f(0, 0, 0);
        float velo_abs = fused_velocity_.norm();
        if(velo_abs > 20.0f){
            float scale = 20.0f / velo_abs;
            fused_velocity_.array() *= scale;
        }
        Eigen::Vector3f measured_position = sensor_object->selected_track_point;

        // Init state, use anchor point
        kalman_filter_->Init(measured_position, fused_velocity_);
    } else {
        // TLOG_WARN << "ELSE last_predict_timestamp_ = " << last_predict_timestamp_;
        // Copy object info
        fused_track_point_ = sensor_object->velo_anchor_point;
        in_fov_bound_ = sensor_object->in_fov_bound;
        fused_acceleration_.setZero();
        // [Fix] 第一帧只记录了位置，并未完成滤波器的初始化，应返回 false
        return false;
    }

    return true;
}

void MultiAnchorMotionFilter::UpdateWithObject(const TrackedObjectConstPtr& track_data,
                               TrackedObjectPtr& new_object) {
    // Judge kalman filter status and update object status
    if (filter_init_) {
        double time_diff = new_object->timestamp - last_predict_timestamp_;
        last_predict_timestamp_ = new_object->timestamp;
        
        MotionFusionWithMeasurement(track_data, new_object, time_diff);
    } else {
        filter_init_ = InitFilter(new_object);
        last_predict_timestamp_ = new_object->timestamp;
        return;
    }

    // Update final state for result out
    UpdateMotionState(new_object);
}

void MultiAnchorMotionFilter::MotionFusionWithMeasurement(const TrackedObjectConstPtr& track_data,
        TrackedObjectPtr& measurement, double time_diff)
{
    // Kalman predict step
    auto predict_pose = kalman_filter_->predict(time_diff);

    // Generate observations from measurements
    motion_meas_->ComputeMotionMeasurment(track_data, measurement);

    Eigen::Vector3f measured_position = measurement->selected_track_point;

    StateChange(track_data, measurement);

    auto ekf_res = kalman_filter_->update(measured_position,
                                        measurement->selected_measured_velocity);
}

void MultiAnchorMotionFilter::UpdateWithoutObject(const TrackedObjectConstPtr& latest_object,
                             TrackedObjectPtr& track_data)
{
    // Update without measurement
    // MotionFusionWithoutMeasurement(time_diff);

    return;
}

void MultiAnchorMotionFilter::UpdateMotionState(TrackedObjectPtr& new_object) {

    fused_track_point_(0) = kalman_filter_->getState()(0);
    fused_track_point_(1) = kalman_filter_->getState()(1);
    fused_velocity_   (0) = kalman_filter_->getState()(2);
    fused_velocity_   (1) = kalman_filter_->getState()(3);
    fused_state_cov_      = kalman_filter_->getCovariance();
    // Update track point
    new_object->output_selected_track_point = fused_track_point_;

    // Update velocity and acceleration
    new_object->output_velocity = fused_velocity_;

    //Update state covariance
    new_object->output_state_covariance = fused_state_cov_;
}

void MultiAnchorMotionFilter::MotionFusionWithoutMeasurement(const double time_diff)
{}

void MultiAnchorMotionFilter::UpdateObjectHistory(const Eigen::Vector3f& velocity,
                                                  const double& timestamp) {

    // Out of stack
    if (object_history_info_.size() > uint(params_.sensor_eval_window_size)) {
        object_history_info_.pop_front();
    }

    // Push to stack
    object_history_info_.emplace_back(timestamp, velocity);
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
