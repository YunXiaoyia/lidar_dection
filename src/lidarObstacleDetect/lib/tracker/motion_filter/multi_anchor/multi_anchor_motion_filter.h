
//
// Created by jie.gong on 22-02-17.
//

#pragma once

#include "../../lidarnetsdk/src/common/toml.hpp"
#include "common/base_track_filter.h"

#include "motion_filter/multi_anchor/multi_anchor_motion_measurement.h"
#include "motion_filter/kalman_model/kalman_filter.h"

#include <Eigen/Dense>
#include <deque>

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {


// For record history data
struct TrackInfo {
    double timestamp;
    Eigen::Vector3f velocity;

    TrackInfo(double _timestamp,
            Eigen::Vector3f _velocity) {
        timestamp   = _timestamp;
        velocity    = _velocity;
    }
};

// Kalman kinematic model
enum class FilterModel : uint8_t {
    EKF_CV,
    EKF_CA,
    EKF_CTRV,
    KF_CV,
};

struct MultiAnchorParams {
    // Sensor evalution window size
    int sensor_eval_window_size = 4.0;

    // Velocity direction change threshold
    float velocity_angle_change_threshold = static_cast<float>(M_PI / 20.0);

    // Acceleration angle change threshold
    float acceleration_angle_change_threshold = static_cast<float>(M_PI / 3.0);

    // Velocity length change threshold  m/s
    float velocity_length_change_threshold = 5.0;

    // Trace back time threshold
    float trace_time_threshold = 2.0;

    // Kalman filter type
    std::string filter_type = "kf_cv";

    MultiAnchorParams() = default;
};

class MultiAnchorMotionFilter : public BaseTrackFilter {
public:
    /**
     * @brief Construct a new Kalman Motion Filter object
     *
     * @param[in] track data of track object
     */
    explicit MultiAnchorMotionFilter() = default;
    ~MultiAnchorMotionFilter() = default;

    MultiAnchorMotionFilter(const MultiAnchorMotionFilter&) = delete;
    MultiAnchorMotionFilter& operator=(const MultiAnchorMotionFilter&) = delete;

    /**
     * @brief init static params
     *
     * @param param_node params handle
     * @return true if success or false if there are something failed
     */
    static bool InitStaticParams(const toml::node_view<const toml::node>& param_node);

    /**
     * @brief init kalman filter and some magic number
     *
     * @return true if success or false if there are something failed
     */
    bool Init(
        const TrackFilterInitOptions& options = TrackFilterInitOptions()) override;

    /**
     * @brief update the tracker with current measurement
     *
     * @param[in] measurement sensor results
     * @param[in] target_timestamp tracker timestamp
     */
    void UpdateWithObject(const TrackedObjectConstPtr& track_data,
                            TrackedObjectPtr& new_object) override;

    /**
     * @brief update the tracker only use time diff
     *
     * @param[in] measurement_timestamp kalman predict update time
     */
    void UpdateWithoutObject(const TrackedObjectConstPtr& latest_object,
                             TrackedObjectPtr& track_data) override;

    void TransState(const Eigen::Isometry3f& tf) override;

    std::string Name() const override { return "MultiAnchorMotionFilter"; }

private:
    bool InitFilter(const TrackedObjectConstPtr& sensor_object);
    void MotionFusionWithoutMeasurement(const double time_diff);
    void UpdateMotionState(TrackedObjectPtr& new_object);
    void MotionFusionWithMeasurement(const TrackedObjectConstPtr& track_data,
                                    TrackedObjectPtr& measurement,
                                    double time_diff);
    void StateChange(const TrackedObjectConstPtr& track_data,
                     TrackedObjectPtr& measurement);

    /**
     * @brief record object history info, sensor_type, velocity, timestamp
     *
     */
    void UpdateObjectHistory(const Eigen::Vector3f& velocity,
                             const double& timestamp);

private:

    static MultiAnchorParams params_;

    // motion measurement
    std::unique_ptr<MultiAnchorMotionMeasurement> motion_meas_;

    // Kalman filter object
    std::unique_ptr<IKalmanFilter> kalman_filter_;

    // Kalman init mark
    bool filter_init_ = false;

    // History object info record
    std::deque<TrackInfo> object_history_info_;

    double last_predict_timestamp_;

    bool in_fov_bound_ = false;

    // For fusion result
    Eigen::Vector3f fused_track_point_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f fused_velocity_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f fused_acceleration_ = Eigen::Vector3f::Zero();
    Eigen::Matrix4f fused_state_cov_ = Eigen::Matrix4f::Identity();
};

PERCEPTION_REGISTER_TRACK_FILTER(MultiAnchorMotionFilter);


}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
