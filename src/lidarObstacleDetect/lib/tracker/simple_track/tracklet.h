//
// Created by jie.gong on 22-03-17.
//
#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <deque>
#include <Eigen/Core>

#include "common/base.h"
#include "../../lidarnetsdk/include/lidar_net/object.h"
#include "../../../lidarnetsdk/src/common/toml.hpp"
#include "common/tracked_object.h"
#include "common/base_track_filter.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

enum class TrackletStatus {
    DELETED,
    SKEPTICAL,
    CONFIRMED
};

struct TrackletInitParams {
    float fov_theta;
    float x_direction_offset;
};

/**
 * @brief
 *
 */
class Tracklet {

private:
    // Detect objects buffer capacity
    static int MAX_OBJECT_BUFFER_NUM;

    // Tracked objects buffer capacity
    static int MAX_TRACK_BUFFER_NUM;

    // Maximum disappearance time
    static double VALID_LOST_DURATION;

    // Maximum disappearance counters
    static int MAX_CONTINUOUS_LOST_NUM;

    // FOV bound distance thresold, for tricks
    static float FOV_BOUND_DIS;

    // Minimun appearance counters
    static int MIN_CONTINUOUS_DET_NUM;

public:
    /**
     * @brief Construct a new Tracklet object
     *
     */
    Tracklet();

    /**
     * @brief init static params
     *
     * @param param_node params handle
     * @return true if success or false if there are something failed
     */
    static bool InitStaticParams(const toml::node_view<const toml::node>& param_node);

    /**
     * @brief Main function is convert a new detected object to a tracked object
     *
     * @param object New detected objects
     * @param init_params
     */
    void Init(double frame_timestamp, const common::ObjectPtr& object, const TrackletInitParams& init_params);

    /**
     * @brief Update current tracked objects with a new measurement, include motion, shape, states...
     *
     * @param object_new
     */
    void updateWithObject(const common::ObjectPtr& object_new);

    /**
     * @brief Excute predict process
     *
     * @param timestamp predict end time
     */
    void updateWithoutObject(const double& timestamp);

    /**
     * @brief update object type
     *
     * @param common::ObjectType& object_type
     */
    void updateObjectType(common::ObjectType& object_type);
    /**
     * @brief Change to global coordinates based on current position, mainly for estimate absolute velocity
     *
     * @param tf Transformation matrix
     */
    void TransformToCurrentFrame(const Eigen::Isometry3f& tf);

    /**
     * @brief Determine whether it is at the boundary of FOV, tricks for velocity
     *
     * @param object
     * @param latest_object
     * @return float
     */
    bool inFovBound(const common::ObjectPtr& object);
    float calFovOffset(const common::ObjectPtr& object, TrackedObjectConstPtr& latest_object);

    const int& getMinContinuousDetNumThreshold(){
        return MIN_CONTINUOUS_DET_NUM;
    }

    /**
     * @brief Get the Latest Tracked Object\Detected object\Out tracked objects
     *
     * @return TrackedObjectConstPtr
     */
    TrackedObjectConstPtr GetLatestTrackedObject();
    common::ObjectPtr GetLatestDetectedObject();
    TrackedObjectConstPtr GetOutTrackedObject();


private:

    void PushTrackedObjectToTrack(const TrackedObjectPtr& new_tracked);
    void PushDetectedObjectToDetect(const common::ObjectPtr& new_detected);

    void updateTrackInfo(TrackedObjectPtr& tracked_object);

private:
    // Only Confirmed tracklet have this id, the value is 0 in other states
    size_t track_id_ = 0;

    // Number of frames surviving in tracking pool, larger means more stable
    size_t track_age_;

    // Number of consecutive undetected frames, for update status, compared with MAX_CONTINUOUS_LOST_NUM
    int consecutive_invisible_count_;

    // Number of consecutive detected frames
    int consecutive_visible_count_;

    // Motion filter handle
    std::shared_ptr<BaseTrackFilter> motion_filter_;

    // Shape filter handle
    std::shared_ptr<BaseTrackFilter> shape_filter_;

    // Detected history objects, cached for other purposes
    std::deque<common::ObjectPtr> history_detected_objects_;

    // Tracked history objects, cached for output and as a priori information for updating the current frame
    std::deque<TrackedObjectPtr> history_tracked_objects_;

    // Tracked and untracked timestamp record
    double last_tracked_time_;
    double last_untracked_time_;

    // Tracklet status, only Confirmed can export out
    TrackletStatus status_ = TrackletStatus::SKEPTICAL;

    // For lidar FOV bound tricks, lidar horizon fov angle, x-distance in car frame
    float fov_tan_theta_;
    float x_correction_offset_;

    // Static parameters
    static std::string SHAPE_FILTER_TYPE;
    static std::string MOTION_FILTER_TYPE;

public:

    const size_t& track_id() const { return track_id_; }
    const size_t& track_age() const { return track_age_; }
    // const double& last_tracked_time() const { return last_tracked_time_; }
    const int& consecutive_invisible_count() const {return consecutive_invisible_count_; }
    const int& consecutive_visible_count() const {return consecutive_visible_count_; }

    const std::deque<common::ObjectPtr>& history_detected_objects() { return history_detected_objects_; }
    const std::deque<TrackedObjectPtr>& history_tracked_objects() { return history_tracked_objects_; };

    inline bool timeValid(const double& cur_time) const
        { return cur_time - last_tracked_time_ < VALID_LOST_DURATION; }
    inline bool isAlive(const double& cur_time) const
        { return timeValid(cur_time) && consecutive_invisible_count_ < MAX_CONTINUOUS_LOST_NUM; };
    inline bool isConfirmed() const { return status_ == TrackletStatus::CONFIRMED; };
    inline bool isSkeptical() const { return status_ == TrackletStatus::SKEPTICAL; };
    inline bool isDeleted() const { return status_ == TrackletStatus::DELETED; };

    void set_status(const TrackletStatus& status) { status_ = status; };
    void set_track_id(const size_t& id) { track_id_ = id; };


};

using TrackletPtr = std::shared_ptr<Tracklet>;
using TrackletConstPtr = std::shared_ptr<const Tracklet>;

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
