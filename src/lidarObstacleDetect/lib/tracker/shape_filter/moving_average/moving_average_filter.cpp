
#include "shape_filter/moving_average/moving_average_filter.h"

#include "Eigen/src/Core/Matrix.h"
#include "common/depthink_time.h"
#include "lib/utils/geometry.hpp"

#include <iostream>

// 临时定义 TLOG_WARN 宏
#ifndef TLOG_WARN
#define TLOG_WARN std::cout << "[WARN] "
#endif

// 临时定义 utils 命名空间和 inFOV 函数
namespace utils {
    inline bool inFOV(float x, float y, float fov_degrees, float delta_x_corner_shift) {
        // 简单的 FOV 检查：检查点是否在指定的视野范围内
        // fov_degrees 是视野角度，delta_x_corner_shift 是 X 方向的偏移
        float fov_rad = fov_degrees * M_PI / 180.0f;
        float tan_half_fov = std::tan(fov_rad / 2.0f);
        
        // 检查 y 方向是否在视野范围内，考虑 x 方向的偏移
        if (x > -delta_x_corner_shift) {
            return std::abs(y) < (x + delta_x_corner_shift) * tan_half_fov;
        }
        return false;
    }
}

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

// Moving average coefficient of heading angle
float MovingAverageShapeFilter::AnglekMovingAverage = 0.2;

// Moving average coefficient of bbox size
float MovingAverageShapeFilter::SizekMovingAverage = 0.2;

float MovingAverageShapeFilter::fov_ = 80.0;

float MovingAverageShapeFilter::delta_x_corner_shift_ = 8.0;

bool MovingAverageShapeFilter::InitStaticParams(const toml::node_view<const toml::node>& param_node) {

    auto angle_moving_average = param_node.at_path("angle_coefficient").value<float>();
    if (!angle_moving_average.has_value()) {
        TLOG_WARN << "[ShapeFilter] Dont't find AnglekMovingAverage definition! please check param: angle_coefficient."
                     << " use default: 0.2.";
    } else {
        AnglekMovingAverage = angle_moving_average.value();
    }

    auto size_moving_average = param_node.at_path("size_coefficient").value<float>();
    if (!angle_moving_average.has_value()) {
        TLOG_WARN << "[ShapeFilter] Dont't find SizekMovingAverage definition! please check param: size_coefficient."
                     << " use default: 0.2.";
    } else {
        SizekMovingAverage = size_moving_average.value();
    }

    auto fov = param_node.at_path("fov").value<float>();
    if (!fov.has_value()) {
        TLOG_WARN << "[ShapeFilter] Dont't find fov definition! please check param: fov."
                     << " use default: " << fov_;
    } else {
        fov_ = fov.value();
    }

    auto delta_x_corner_shift = param_node.at_path("delta_x_corner_shift").value<float>();
    if (!delta_x_corner_shift.has_value()) {
        TLOG_WARN << "[ShapeFilter] Dont't find delta_x_corner_shift definition! please check param: delta_x_corner_shift."
                     << " use default: " << delta_x_corner_shift_;
    } else {
        delta_x_corner_shift_ = delta_x_corner_shift.value();
    }

    return true;
}


bool MovingAverageShapeFilter::Init(
      const TrackFilterInitOptions& options) {

  // TODO: 初始化操作

  return true;
}

void MovingAverageShapeFilter::UpdateWithObject(const TrackedObjectConstPtr& latest_object,
                                      TrackedObjectPtr& new_object) {

    const auto& latest_direction = latest_object->output_direction;
    const auto& latest_theta = latest_object->output_theta;
    const auto& new_direction = new_object->direction;
    // const auto& new_theta = new_object->theta; // error: unused variable

    auto& output_direction = new_object->output_direction;
    auto& output_theta = new_object->output_theta;

    // Avoid direction reverse. pi
    if (new_direction.dot(latest_direction) < 0) {
        output_direction = (-1) * new_direction;
        output_theta = std::atan2(output_direction.y(), output_direction.x());
    }

    // Avoid direction exchange. pi/2
    if (fabs(fabs(latest_theta) - fabs(output_theta)) > M_PI_4f32) {

        if (latest_theta > output_theta) {
            output_theta += M_PI_2f32;
        } else {
            output_theta -= M_PI_2f32;
        }

        output_direction.x() = std::cos(output_theta);
        output_direction.y() = std::sin(output_theta);
        output_direction.z() = 0.0f;
    }

    // Update direction
    output_direction = latest_direction * (1 - AnglekMovingAverage) + output_direction * AnglekMovingAverage;
    output_direction.normalize();

    // Update size
    float offset_x = std::numeric_limits<float>::max();
    float offset_y = std::numeric_limits<float>::max();
    for (int i = 0; i < 4; ++i) {
        offset_x = std::min(offset_x, new_object->corners[i].x());
        offset_y = std::min(offset_y, new_object->corners[i].y());
    }
    Eigen::Vector2f offset(offset_x, offset_y);

    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    Eigen::Vector2f direction = output_direction.head<2>();
    Eigen::Vector2f odirection(-direction(1), direction(0));

    for(int i = 0; i < new_object->object_ptr->bbox.corners2d.cols(); ++i){
        Eigen::Vector2f polygon_xy;
        polygon_xy << new_object->object_ptr->bbox.corners2d(0, i), new_object->object_ptr->bbox.corners2d(1, i);
        polygon_xy -= offset;
        float projected_x = direction.dot(polygon_xy);
        float projected_y = odirection.dot(polygon_xy);
        min_x = std::min(min_x, projected_x);
        max_x = std::max(max_x, projected_x);

        min_y = std::min(min_y, projected_y);
        max_y = std::max(max_y, projected_y);
    }
    new_object->output_size[0] = max_x - min_x;
    new_object->output_size[1] = max_y - min_y;

    // 判断是否在FOV边界处
    Eigen::Vector2f fov_anchor;
    if(new_object->center.y() > 0){
        // 左侧，右下角
        fov_anchor.x() =  -new_object->size.x()/2.0f * direction(0) - new_object->size.y()/2.0f * odirection(0) +
                                        new_object->output_center.x();
        fov_anchor.y() =  -new_object->size.x()/2.0f  * direction(1) - new_object->size.y()/2.0f * odirection(1) +
                                        new_object->output_center.y();
    }
    else{
        // 右侧，看左下角
        fov_anchor.x() = -new_object->size.x()/2.0f * direction(0) + new_object->size.y()/2.0f * odirection(0) +
                                            new_object->output_center.x();
        fov_anchor.y() = -new_object->size.x()/2.0f * direction(1) + new_object->size.y()/2.0f * odirection(1) +
                                        new_object->output_center.y();
    }

    // update box size
    new_object->output_size = latest_object->output_size * (1 - SizekMovingAverage) + new_object->output_size * SizekMovingAverage;
    float new_length =  new_object->output_size.x();
    float new_half_width = new_object->output_size.y() / 2.0;

    // 判断是否在边界处，false为在边界
    bool flag_in_fov = utils::inFOV(fov_anchor.x(), fov_anchor.y(), fov_, delta_x_corner_shift_);
    if(flag_in_fov){
        // true, 使用tail_center作为shape跟踪点
        Eigen::Vector2f projected_tail_xy;
        projected_tail_xy << min_x, (max_y + min_y) * 0.5;

        Eigen::Vector2f tail_center_xy = projected_tail_xy(0) * direction +
                                  projected_tail_xy(1) * odirection + offset;
        // Update corners     0--3
        //                    |  |
        //                    1--2
        new_object->output_corners[3].x() = new_length * direction(0) + new_half_width * odirection(0) +
                                            tail_center_xy.x();
        new_object->output_corners[3].y() = new_length * direction(1) + new_half_width * odirection(1) +
                                            tail_center_xy.y();
        new_object->output_corners[2].x() = new_length * direction(0) - new_half_width * odirection(0) +
                                            tail_center_xy.x();
        new_object->output_corners[2].y() = new_length * direction(1) - new_half_width * odirection(1) +
                                        tail_center_xy.y();

        new_object->output_corners[1].x() = - new_half_width * odirection(0) + tail_center_xy.x();
        new_object->output_corners[1].y() = - new_half_width * odirection(1) + tail_center_xy.y();

        new_object->output_corners[0].x() = new_half_width * odirection(0) + tail_center_xy.x();
        new_object->output_corners[0].y() = new_half_width * odirection(1) + tail_center_xy.y();
    }
    else{
        // false，使用head_center作为shape跟踪点
        Eigen::Vector2f projected_head_xy;
        projected_head_xy << max_x, (max_y + min_y) * 0.5;

        Eigen::Vector2f head_center_xy = projected_head_xy(0) * direction +
                                  projected_head_xy(1) * odirection + offset;
        // Update corners     0--3
        //                    |  |
        //                    1--2
        new_object->output_corners[3].x() = new_half_width * odirection(0) + head_center_xy.x();
        new_object->output_corners[3].y() = new_half_width * odirection(1) + head_center_xy.y();


        new_object->output_corners[2].x() = - new_half_width * odirection(0) + head_center_xy.x();
        new_object->output_corners[2].y() = - new_half_width * odirection(1) + head_center_xy.y();

        new_object->output_corners[1].x() = -(new_length * direction(0)) - new_half_width * odirection(0) +
                                            head_center_xy.x();
        new_object->output_corners[1].y() = -(new_length * direction(1)) - new_half_width * odirection(1) +
                                            head_center_xy.y();

        new_object->output_corners[0].x() =  -(new_length * direction(0)) + new_half_width * odirection(0) +
                                            head_center_xy.x();
        new_object->output_corners[0].y() = -(new_length * direction(1)) + new_half_width * odirection(1) +
                                            head_center_xy.y();
    }

    // center
    new_object->output_center.x() = (new_object->output_corners[0].x() + new_object->output_corners[1].x() +
                                    new_object->output_corners[2].x() + new_object->output_corners[3].x()) / 4.0f;
    new_object->output_center.y() = (new_object->output_corners[0].y() + new_object->output_corners[1].y() +
                                    new_object->output_corners[2].y() + new_object->output_corners[3].y()) / 4.0f;

    // Update theta
    new_object->output_theta = std::atan2(output_direction.y(), output_direction.x());

}

void MovingAverageShapeFilter::UpdateWithoutObject(const TrackedObjectConstPtr& latest_object,
                             TrackedObjectPtr& track_data) {
  // TODO(.)
}


}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

// 注册 MovingAverageShapeFilter
namespace {
    struct MovingAverageShapeFilter_Registrar {
        MovingAverageShapeFilter_Registrar() {
            highway::perception::Registerer<highway::perception::track::BaseTrackFilter>::Instance().Register("MovingAverageShapeFilter", 
                []() -> std::shared_ptr<highway::perception::track::BaseTrackFilter> { 
                    return std::make_shared<highway::perception::track::MovingAverageShapeFilter>(); 
                });
        }
    };
    static MovingAverageShapeFilter_Registrar g_MovingAverageShapeFilter_registrar;
}
