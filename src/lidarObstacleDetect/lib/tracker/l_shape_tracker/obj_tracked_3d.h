/**
 * @file obj_tracked_3d.h
 * @author xxx (xxx@depthink.tech)
 * @brief
 * @version 0.1
 * @date 2021-06-29
 *
 * @copyright Copyright (c) 2021 depthink
 *
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <deque>

#include "common/base.h"

// #include "data_manager.h"
#include "l_shape_filter.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

/**
 * @brief
 */
class ObjTracked3d {
   public:
    ObjTracked3d();
    ObjTracked3d(const ObjTracked3d& track);
    ObjTracked3d& operator=(const ObjTracked3d& track);

    /**
     * @brief
     *
     * @param tf
     * @param stamp
     */
    void predict(const Eigen::Isometry3f& tf, const double& stamp);

    /**
     * @brief
     *
     * @param dt
     */
    void update(const common::ObjectPtr& detect, const double& cur_time);

    /// @brief
    inline bool dieout() {
        return consecutive_lost_ >= 5 || (life_time_ < 5 && consecutive_lost_ >= 2);
        //  || (life_time > 20 && obj_type == common::ObjectType::UNKNOWN); // ---------------
    };

   public:
    uint16_t track_id_ = 0u;        ///<
    int16_t life_time_ = 0;         ///<
    int16_t consecutive_lost_ = 0;  ///<

    std::deque<common::ObjectType> type_deq_;            ///<
    common::ObjectType obj_type_ = common::ObjectType::UNKNOWN;  ///<

    common::ObjectPtr last_detect_obj_;             ///<
    LshapeFilter l_shape_filter_; 
    Eigen::Matrix4f state_cov_ = Eigen::Matrix4f::Identity();
    Eigen::Vector2f reference_point_;          /// tracking点          
    Eigen::Vector3f center_;                  ///<
    Eigen::Vector3f velocity_;                ///< 跟踪的速度
    std::array<Eigen::Vector3f, 4> corners_;  ///< 角点
    Eigen::Vector3f size_ = Eigen::Vector3f::Zero();                  ///<
    float direction_ = 0.0f;

    // History tracked object buffer 
    std::deque<common::ObjectPtr> object_tracked_buffer_;  

    // History detected object buffer
    std::deque<common::ObjectPtr> object_detected_buffer_;  

    float vx_ = 0;

};

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
