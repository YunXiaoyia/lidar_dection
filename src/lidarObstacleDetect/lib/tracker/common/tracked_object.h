
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <Eigen/src/Core/Matrix.h>
#include "common/base.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

enum class ReferencePointType : std::uint8_t {
    L_CORNER,
    BARY_CENTER,
    BBOX_CENTER,
    TAIL_MIDDLE,
    TAIL_SIDE
};

Eigen::Vector3f calTailMiddlePoint(const common::BoundingBox& bbox);

struct TrackedObject {
    TrackedObject();
    TrackedObject& operator=(const TrackedObject& rhs) = default;

    void Reset();

    void AttachObject(
        common::ObjectPtr obj_ptr,
        const Eigen::Vector3d& global_to_local_offset = Eigen::Vector3d::Zero());

    void ToObject(common::ObjectPtr& obj_ptr) const;

    bool isTailMiddlePoint() const;

    static ReferencePointType TRACK_POINT_TYPE;

    double timestamp = -1;

    // @brief track id, required
    int track_id = -1;

    // @brief timestamp of latest measurement, required
    // double latest_tracked_time = 0.0;

    // Corner case, in fov bound
    bool in_fov_bound = false;
    // float fov_offset = 0.0;

    // ***************************************************
    // self information from match
    // ***************************************************
    // std::vector<float> shape_features;
    // std::vector<float> shape_features_full;
    // size_t histogram_bin_size = 10;
    // association distance
    // range from 0 to max_match_distance
    float association_score = 0.0f;

    // ***************************************************
    // measurement correlative information from object_ptr
    // ***************************************************
    common::ObjectPtr object_ptr;
    // corners always store follow const order based on object direction
    Eigen::Vector3f corners[4];  // TODO: 需确定在那个地方计算这个信息，measurement_collection中有用
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    // Eigen::Vector3f barycenter;
    Eigen::Vector3f anchor_point = Eigen::Vector3f::Zero();  //  barycenter
    // bbox size
    Eigen::Vector3f size = Eigen::Vector3f::Zero();

    Eigen::Vector3f direction = Eigen::Vector3f::Zero();

    Eigen::Vector3f velo_anchor_point = Eigen::Vector3f::Zero();

    float theta = 0.0f;

    Eigen::Vector3f selected_track_point = Eigen::Vector3f::Zero();

    // ***************************************************
    // measurement correlative information from measurement computer
    // ***************************************************
    Eigen::Vector3f measured_barycenter_velocity = Eigen::Vector3f::Zero();
    Eigen::Vector3f measured_center_velocity = Eigen::Vector3f::Zero();
    Eigen::Vector3f measured_detect_bbox_center_velocity = Eigen::Vector3f::Zero();
    Eigen::Vector3f measured_nearest_corner_velocity = Eigen::Vector3f::Zero();  // no projection
    Eigen::Vector3f measured_corners_velocity[4];

    // ***************************************************
    // filter correlative information
    // ***************************************************
    // states
    // double update_quality = 0.0;
    Eigen::Vector3f selected_measured_velocity = Eigen::Vector3f::Zero();
    // Eigen::Vector3f selected_measured_acceleration;

    // ***************************************************
    // postprocess correlative information
    // ***************************************************
    Eigen::Vector3f output_velocity = Eigen::Vector3f::Zero();
    Eigen::Matrix4f output_state_covariance = Eigen::Matrix4f::Identity();
    // Eigen::Matrix3f output_velocity_uncertainty;
    Eigen::Vector3f output_direction = Eigen::Vector3f::Zero();
    Eigen::Vector3f output_center = Eigen::Vector3f::Zero();
    Eigen::Vector3f output_size = Eigen::Vector3f::Zero();
    common::ObjectType output_type = common::ObjectType::UNKNOWN;
    Eigen::Vector3f output_corners[4];
    float output_theta = 0.0f;

    Eigen::Vector3f output_selected_track_point = Eigen::Vector3f::Zero();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;  // struct TrackedObject

typedef std::shared_ptr<TrackedObject> TrackedObjectPtr;
typedef std::shared_ptr<const TrackedObject> TrackedObjectConstPtr;

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

