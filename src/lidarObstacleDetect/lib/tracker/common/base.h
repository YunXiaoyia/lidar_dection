#pragma once
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Macros.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {
namespace common {

// 定义点云类型
using PointType = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointType>;
using PointCloudPtr = PointCloud::Ptr;

// 目标类型 (参考 l_shape_track.cpp 中的逻辑)
enum class ObjectType : uint8_t {
    UNKNOWN = 0,
    VEHICLE = 1,
    PEDESTRIAN = 2,
    CYCLIST = 3,
    BARRIER = 4,
    TYPE_NUM = 5
};

// L-Shape 特征
struct LShapeBox {
    Eigen::Vector3d reference_point; // (x, y, 0)
    Eigen::Vector3d l_shape;         // (L1, L2, theta)
};

// 包围盒
struct BoundingBox {
    Eigen::Vector3f center;
    Eigen::Vector3f size;
    Eigen::Matrix<float, 2, 4> corners2d;
    float theta;
};

// 追踪目标对象
struct Object {
    int track_id = -1;
    int detect_id = -1; // 对应 detector 的 ID
    double timestamp = 0.0;
    int age = 0; // life_time

    ObjectType type = ObjectType::UNKNOWN;

    BoundingBox bbox;
    LShapeBox lshape_box;

    Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
    Eigen::Matrix4f state_covariance = Eigen::Matrix4f::Identity();

    PointCloudPtr points; 

    Object() {
        points.reset(new PointCloud());
    }
};

using ObjectPtr = std::shared_ptr<Object>;

// 帧数据
struct LidarFrame {
    double timestamp;
    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
    std::vector<ObjectPtr> detected_objects;
    std::vector<ObjectPtr> tracked_objects;
};

using LidarFramePtr = std::shared_ptr<LidarFrame>;

} // namespace common
} // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
