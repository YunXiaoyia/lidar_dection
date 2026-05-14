#include "common/tracked_object.h"
#include <limits>
#include "Eigen/src/Core/Matrix.h"
#include "common/lidar_perception_log.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

ReferencePointType TrackedObject::TRACK_POINT_TYPE = ReferencePointType::TAIL_MIDDLE;

Eigen::Vector3f calTailMiddlePoint(const common::BoundingBox& bbox) {

    Eigen::Vector3f res = Eigen::Vector3f(0, 0, 0);

    std::vector<Eigen::Vector3f> corners;
    for (int i = 0; i < 4; ++i){
        corners.emplace_back(Eigen::Vector3f(bbox.corners2d(0, i),
                                            bbox.corners2d(1, i),
                                            0.0));
    }
    std::sort(corners.begin(), corners.end(), [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b){
        return (a.x() < b.x());
    });
    res = (corners[0] + corners[1]).array() / 2;

    return res;

}

Eigen::Vector3f calTailSidePoint(const Eigen::Vector3f& size_in,
                                 const Eigen::Vector3f& center_in,
                                 const Eigen::Vector3f& direction_in) {

    Eigen::Vector3f res_out = Eigen::Vector3f::Zero();

    Eigen::Vector2f direction = direction_in.head<2>();
    Eigen::Vector2f odirection(-direction(1), direction(0));

    float half_length = size_in.x() * 0.5;
    float half_width = size_in.y() * 0.5;

    if (center_in.x() < 20) {
        if (center_in.y() < 0) {
            res_out.x() = -half_length * direction(0) - half_width * odirection(0) +
                                         center_in.x();
            res_out.y() = -half_length * direction(1) - half_width * odirection(1) +
                                         center_in.y();
        } else {
            res_out.x() = -half_length * direction(0) + half_width * odirection(0) +
                                         center_in.x();
            res_out.y() = -half_length * direction(1) + half_width * odirection(1) +
                                         center_in.y();
        }
    } else {
        if (center_in.y() < 0) {
            res_out.x() = half_length * direction(0) - half_width * odirection(0) +
                                         center_in.x();
            res_out.y() = half_length * direction(1) - half_width * odirection(1) +
                                         center_in.y();
        } else {
            res_out.x() = half_length * direction(0) + half_width * odirection(0) +
                                         center_in.x();
            res_out.y() = half_length * direction(1) + half_width * odirection(1) +
                                         center_in.y();
        }
    }
    return res_out;
}

Eigen::Vector3f calClosetPoint(Eigen::Vector3f* corners) {
    Eigen::Vector3f origin(20.0f, 0.0f, 0.0f);

    float min_dist = std::numeric_limits<float>::max();
    int min_idx = 0;
    for (int i = 0; i < 4; ++i) {

        float dist = (corners[i] - origin).head(2).norm();

        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }

    return corners[min_idx];
}

TrackedObject::TrackedObject() {
    measured_corners_velocity[0] = Eigen::Vector3f::Zero();
    measured_corners_velocity[1] = Eigen::Vector3f::Zero();
    measured_corners_velocity[2] = Eigen::Vector3f::Zero();
    measured_corners_velocity[3] = Eigen::Vector3f::Zero();

    corners[0] = Eigen::Vector3f::Zero();
    corners[1] = Eigen::Vector3f::Zero();
    corners[2] = Eigen::Vector3f::Zero();
    corners[3] = Eigen::Vector3f::Zero();

    output_corners[0] = Eigen::Vector3f::Zero();
    output_corners[1] = Eigen::Vector3f::Zero();
    output_corners[2] = Eigen::Vector3f::Zero();
    output_corners[3] = Eigen::Vector3f::Zero();
}

void TrackedObject::AttachObject(common::ObjectPtr obj_ptr,
                                 const Eigen::Vector3d& global_to_local_offset) {
    if (obj_ptr == nullptr) { return; }

    // all state of input obj_ptr will not change except cloud world
    object_ptr = obj_ptr;

    timestamp = object_ptr->timestamp;

    // [关键修复开始] ==========================================
    // 1. 先判断 points 是否为空指针
    if (object_ptr->points == nullptr) { 
        return; 
    }

    // 2. 再判断 points 是否有数据 (size > 0)
    if (object_ptr->points->empty()) {
        // [修复核心] 如果点云为空（删除了resize后会进入这里）
        // 直接使用 BBox 的中心作为锚点。
        // 这是正确的逻辑：没有点云细节时，就信赖检测框的位置。
        anchor_point = object_ptr->bbox.center;
    } else {
        // 只有当真的有点云数据时，才去算 Mean
        // 这样就避开了 Eigen 的 Assertion 崩溃
        auto points_eigen = object_ptr->points->getMatrixXfMap(3, 8, 0);
        anchor_point = points_eigen.rowwise().mean();
    }
    // [关键修复结束] ==========================================

    size = object_ptr->bbox.size;

    center = object_ptr->bbox.center;

    theta = object_ptr->bbox.theta;

    direction.x() = std::cos(object_ptr->bbox.theta);
    direction.y() = std::sin(object_ptr->bbox.theta);
    direction.z() = 0.0f;

    if (object_ptr->bbox.corners2d.rows() == 0 || object_ptr->bbox.corners2d.cols() == 0) {
        return;
    }
    
    corners[0].head(2) = object_ptr->bbox.corners2d.col(0);
    corners[1].head(2) = object_ptr->bbox.corners2d.col(1);
    corners[2].head(2) = object_ptr->bbox.corners2d.col(2);
    corners[3].head(2) = object_ptr->bbox.corners2d.col(3);

    if(TRACK_POINT_TYPE == ReferencePointType::BARY_CENTER) {
        selected_track_point = anchor_point;
    } else if (TRACK_POINT_TYPE == ReferencePointType::BBOX_CENTER) {
        selected_track_point = center;
    } else if (TRACK_POINT_TYPE == ReferencePointType::TAIL_MIDDLE) {
        selected_track_point = calTailMiddlePoint(object_ptr->bbox);
    } else if (TRACK_POINT_TYPE == ReferencePointType::L_CORNER) {
        // selected_track_point = calTailMiddlePoint(object_ptr->bbox);
    } else if (TRACK_POINT_TYPE == ReferencePointType::TAIL_SIDE) {
        selected_track_point = calTailSidePoint(object_ptr->bbox.size,
                                                object_ptr->bbox.center, direction);
    } else {
        selected_track_point = anchor_point;
    }

    // velo_anchor_point = calTailSidePoint(object_ptr->bbox.size,
    //                                      object_ptr->bbox.center, direction);
    // velo_anchor_point.head(2) = object_ptr->lshape_box.reference_point.cast<float>();
    // velo_anchor_point = calClosetPoint(corners);
    velo_anchor_point = center;

    if (in_fov_bound && isTailMiddlePoint()) {
        selected_track_point.x() = selected_track_point.x() + size.x();

        velo_anchor_point = selected_track_point;
    }

    // TLOG_INFO << "ID:" << track_id << "  in fov:" << in_fov_bound
    //           << " l:" << object_ptr->bbox.size.x() << " w:" << object_ptr->bbox.size.y() << " theta:" << theta;
    // TLOG_INFO << "selected_point:" << selected_track_point.x() << " " << selected_track_point.y();
    // TLOG_INFO << "SELECTED_VELO_POINT " << velo_anchor_point.x() << " " << velo_anchor_point.y();
    // TLOG_INFO << "l-shape:" << object_ptr->lshape_box.reference_point.x() << " " << object_ptr->lshape_box.reference_point.y();
    // TLOG_INFO << corners[0].x() << " " << corners[0].y();
    // TLOG_INFO << corners[1].x() << " " << corners[1].y();
    // TLOG_INFO << corners[2].x() << " " << corners[2].y();
    // TLOG_INFO << corners[3].x() << " " << corners[3].y();

    // Update in motion filter module
    output_velocity = Eigen::Vector3f::Zero();

    // Update in shape filter module
    output_size = object_ptr->bbox.size;
    output_center = object_ptr->bbox.center;
    output_corners[0] = corners[0];
    output_corners[1] = corners[1];
    output_corners[2] = corners[2];
    output_corners[3] = corners[3];
    output_direction  = direction;
    output_theta = theta;

    // TODO: add type filter
    output_type = object_ptr->type;

}

void TrackedObject::Reset() {

}

void TrackedObject::ToObject(common::ObjectPtr& obj) const {

}

bool TrackedObject::isTailMiddlePoint() const {
    return TRACK_POINT_TYPE == ReferencePointType::TAIL_MIDDLE;
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
