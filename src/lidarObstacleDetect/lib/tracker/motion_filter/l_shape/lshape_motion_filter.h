#pragma once

/* Headers for interface class */
#include "common/base_track_filter.h"

/* Headers for shape filter */
#include "lib/utils/kalman/kinematic_model/shape_filter_model.hpp"
#include "lib/utils/kalman/SimpleKalmanFilter.hpp"

/* Headers for motion filter */
#include "motion_filter/kalman_model/kalman_filter.h"

/* Headers for params load*/
#include "../../lidarnetsdk/src/common/toml.hpp"

/* Headers for thirdy party */
#include <Eigen/Dense>

/* Headers for STL library */
#include <deque>
#include <memory>

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {


class LShapeMotionFilter : public BaseTrackFilter {
public:
    /**
     * @brief Construct a new L-shape Kalman Motion Filter object
     * 
     */
    explicit LShapeMotionFilter() = default;
    ~LShapeMotionFilter() = default;

    LShapeMotionFilter(const LShapeMotionFilter&) = delete;
    LShapeMotionFilter& operator=(const LShapeMotionFilter&) = delete;

    /**
     * @brief init static params
     * 
     * @param param_node params handle
     * @return true if success or false if there are something failed
     */
    static bool InitStaticParams(const toml::node_view<const toml::node>& param_node);

    /**
     * @brief init options and some magic number
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

    std::string Name() const override { return "LShapeMotionFilter"; }

private:
    void ClockwisePointSwitch();
    void CounterClockwisePointSwitch();
    double findTurn(const double& new_angle, const double& old_angle);
    int detectCornerPointSwitch(const double& from, const double& to);

    // Export current state
    void ExportStateOut(TrackedObjectPtr& new_object);

    // Check orientation and size
    void findOrientation(float& direction, float& length, float& width);

    // Kalman motion filter object
    std::unique_ptr<IKalmanFilter> motion_filter_;

    // Kalman shape filter object
    std::unique_ptr<Kalman::SimpleShapeKalmanFilter> shape_filter_;
    Kalman::shape::SystemModel shape_sys_model_;
    Kalman::shape::MeasurementModel shape_meas_model_;

    // History theta info
    float last_theta_;
    double last_predict_timestamp_motion_;
    double last_predict_timestamp_shape_;
};

PERCEPTION_REGISTER_TRACK_FILTER(LShapeMotionFilter);
}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
