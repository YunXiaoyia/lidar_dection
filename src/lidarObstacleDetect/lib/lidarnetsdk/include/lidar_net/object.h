#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>

#include <Eigen/Dense>

namespace lidar_net {
namespace base {

/**
 * @brief 目标类型枚举类
 */
enum class ObjectType : std::uint8_t {
    UNKNOWN = 0,  ///< 未知; rule-based 方法结果
    /********************************************************/
    VEHICLE = 1,     ///< 车
    CYCLIST = 2,     ///< 自行车、摩托车
    PEDESTRIAN = 3,  ///< 行人
    BUS = 4,         ///< 大巴车
    CONE = 5,        ///< 锥桶
    BARREL = 6,      ///< 桶
    BARRIER = 7,     ///< 障碍物/护栏
    /********************************************************/
    ART = 10,               ///< ART (通用类型)
    ART_NO_TRAILER = 11,    ///< 不带箱ART
    ART_SEMI_TRAILER = 12,  ///< 带20吋箱ART
    ART_FULL_TRAILER = 13,  ///< 带40吋箱ART
    /********************************************************/
    TRUCK = 20,               ///< 卡车 (通用类型)
    TRUCK_HEAD = 21,          ///< 卡车头
    TRAILER = 22,             ///< 挂车
    TRUCK_NO_TRAILER = 23,    ///< 不带箱挂车
    TRUCK_SEMI_TRAILER = 24,  ///< 带20吋箱挂车
    TRUCK_FULL_TRAILER = 25,  ///< 带40吋箱挂车
    /********************************************************/
    ALIEN_VEHICLE = 30,         ///< 异形车 (通用类型)
    FORKLIFT = 31,              ///< 叉车
    ECCENTRIC_TRUCK_HEAD = 32,  ///< 偏头车
    MOBILE_CRANE = 33,          ///< 流机
    /********************************************************/
};

static std::map<std::string, ObjectType> ObjectTypeDict = {
    {"UNKNOWN", ObjectType::UNKNOWN},
    /********************************************************/
    {"VEHICLE", ObjectType::VEHICLE},
    {"CYCLIST", ObjectType::CYCLIST},
    {"PEDESTRIAN", ObjectType::PEDESTRIAN},
    {"BUS", ObjectType::BUS},
    /********************************************************/
    {"ART", ObjectType::ART},
    {"ART_NO_TRAILER", ObjectType::ART_NO_TRAILER},
    {"ART_SEMI_TRAILER", ObjectType::ART_SEMI_TRAILER},
    {"ART_FULL_TRAILER", ObjectType::ART_FULL_TRAILER},
    /********************************************************/
    {"TRUCK", ObjectType::TRUCK},
    {"TRUCK_HEAD", ObjectType::TRUCK_HEAD},
    {"TRAILER", ObjectType::TRAILER},
    {"TRUCK_NO_TRAILER", ObjectType::TRUCK_NO_TRAILER},
    {"TRUCK_SEMI_TRAILER", ObjectType::TRUCK_SEMI_TRAILER},
    {"TRUCK_FULL_TRAILER", ObjectType::TRUCK_FULL_TRAILER},
    /********************************************************/
    {"ALIEN_VEHICLE", ObjectType::ALIEN_VEHICLE},
    {"FORKLIFT", ObjectType::FORKLIFT},
    {"ECCENTRIC_TRUCK_HEAD", ObjectType::ECCENTRIC_TRUCK_HEAD},
    {"MOBILE_CRANE", ObjectType::MOBILE_CRANE},
    /********************************************************/
    {"CONE", ObjectType::CONE},
    {"BARREL", ObjectType::BARREL},
    {"BARRIER", ObjectType::BARRIER},
    {"barrier", ObjectType::BARRIER},
};

static std::unordered_map<ObjectType, std::string> ObjectEnumStrDict {
    {ObjectType::UNKNOWN, "UNKNOWN"},
    /********************************************************/
    {ObjectType::VEHICLE, "VEHICLE"},
    {ObjectType::CYCLIST, "CYCLIST"},
    {ObjectType::PEDESTRIAN, "PEDESTRIAN"},
    {ObjectType::BUS, "BUS"},
    /********************************************************/
    {ObjectType::ART, "ART"},
    {ObjectType::ART_NO_TRAILER, "ART_NO_TRAILER"},
    {ObjectType::ART_SEMI_TRAILER, "ART_SEMI_TRAILER"},
    {ObjectType::ART_FULL_TRAILER, "ART_FULL_TRAILER"},
    /********************************************************/
    {ObjectType::TRUCK, "TRUCK"},
    {ObjectType::TRUCK_HEAD, "TRUCK_HEAD"},
    {ObjectType::TRAILER, "TRAILER"},
    {ObjectType::TRUCK_NO_TRAILER, "TRUCK_NO_TRAILER"},
    {ObjectType::TRUCK_SEMI_TRAILER, "TRUCK_SEMI_TRAILER"},
    {ObjectType::TRUCK_FULL_TRAILER, "TRUCK_FULL_TRAILER"},
    /********************************************************/
    {ObjectType::ALIEN_VEHICLE, "ALIEN_VEHICLE"},
    {ObjectType::FORKLIFT, "FORKLIFT"},
    {ObjectType::ECCENTRIC_TRUCK_HEAD, "ECCENTRIC_TRUCK_HEAD"},
    {ObjectType::MOBILE_CRANE, "MOBILE_CRANE"},
    /********************************************************/
    {ObjectType::CONE, "CONE"},
    {ObjectType::BARREL, "BARREL"},
    {ObjectType::BARRIER, "BARRIER"},
};

struct BoundingBox {
    using NonAlignVector3f = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;

    // Main direction of the object, required
    NonAlignVector3f direction = NonAlignVector3f(1, 0, 0);

    // The yaw angle
    float theta = 0.0f;

    // Center of the boundingbox (cx, cy, cz), required
    NonAlignVector3f center = NonAlignVector3f(0, 0, 0);  // 对于radar，center被赋值为anchor_point

    // Size = [length, width, height] of boundingbox
    NonAlignVector3f size = NonAlignVector3f(0, 0, 0);  // 分别是与x、y轴平行的长度

    // Probability for each type, required
    Eigen::Matrix<float, -1, 1, Eigen::DontAlign> type_probs;

    // Existence confidence
    float confidence = 0.0;

    // IOU
    float iou = 0.0;

    // Confidence rectified by IOU
    float confidence_iou = 0.0;

    // Contour points
    Eigen::Matrix<float, 2, 4, Eigen::DontAlign> corners2d = Eigen::Matrix<float, 2, 4, Eigen::DontAlign>::Zero();

    // Object type, required
    ObjectType type = ObjectType::UNKNOWN;
};

using BBoxPtr = std::shared_ptr<BoundingBox>;
using BBoxConstPtr = std::shared_ptr<const BoundingBox>;
using BBoxUniPtr = std::unique_ptr<BoundingBox>;
using BBoxConstUniPtr = std::unique_ptr<const BoundingBox>;

}  // namespace base
}  // namespace lidar_net
