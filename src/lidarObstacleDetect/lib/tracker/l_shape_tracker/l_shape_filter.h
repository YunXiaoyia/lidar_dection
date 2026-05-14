/**
 * @file l_shape_tracker.h
 * @authors xxx(fandongsheng@depthink.tech) xxx (xxx@depthink.tech)
 * @brief 
 * @version 0.1
 * @date 2021-04-25
 * 
 * @copyright Copyright (c) 2021 depthink
 * 
 */

#pragma once

#include <Eigen/Dense>
#include <vector>

#include "kalman.h"

using namespace Eigen;

typedef Eigen::Matrix<double, 6, 1> Vector6d;

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

/**
 * @brief 
 * 
 * @param angle 
 * @return double 
 */
double normalize_angle_positive(double angle);

/**
 * @brief 
 * 
 * @param angle 
 * @return double 
 */
double normalize_angle(double angle);

/**
 * @brief 
 * 
 * @param from 
 * @param to 
 * @return double 
 */
double shortest_angular_distance(double from, double to);

/**
 * @brief
 */
class LshapeFilter {
   public:
    LshapeFilter();
    LshapeFilter(const LshapeFilter&);
    LshapeFilter& operator=(const LshapeFilter&);

    /**
     * @brief Construct a new Lshape Tracker object
     *
     * @param x_corner
     * @param y_corner
     * @param L1
     * @param L2
     * @param theta
     * @param cur_time
     */
    LshapeFilter(const double& x_corner, const double& y_corner, const double& L1, const double& L2,
                  const double& theta, const double& cur_time);

    /**
     * @brief
     *
     * @param cur_time
     */
    void predict(const double& cur_time);

    /**
     * @brief
     *
     * @param x_corner
     * @param y_corner
     * @param L1
     * @param L2
     * @param thetaL1
     * @param cur_time
     */
    void update(const double& x_corner, const double& y_corner, const double& L1, const double& L2,
                const double& thetaL1, const double& cur_time);

    /**
     * @brief
     *
     * @param tf
     */
    void transformToCurrent(const Eigen::Isometry3d& tf);

    /**
     * @brief
     *
     * @param center
     * @param velocity
     * @param corner_point_vecs
     * @param extent
     */
    void BoxModel(Eigen::Vector3f& center, Eigen::Vector3f& velocity, std::array<Eigen::Vector3f, 4>& corner_point_vecs,
                  Eigen::Vector3f& extent, float& direction);

    Eigen::VectorXf getMotionState();

    void findOrientation(float& direction, float& length, float& width);

   public:
    KalmanFilter shape_kf;    ///<
    KalmanFilter dynamic_kf;  ///<

    Eigen::Vector2d history_dynamic_meas_;
    Eigen::Vector3d history_shape_meas_;

   private:
    double thetaL1_old;        ///<
    double switchCornerAngle;  ///< angle threshold

    /// @brief
    void ClockwisePointSwitch();

    /// @brief
    void CounterClockwisePointSwitch();

    /**
     * @brief
     *
     * @param new_angle
     * @param old_angle
     * @return double
     */
    double findTurn(const double& new_angle, const double& old_angle);

    /**
     * @brief
     *
     * @param from
     * @param to
     * @return int
     */
    int detectCornerPointSwitch(const double& from, const double& to);

private:
    // last update time
    double last_tracked_time_ = 0.0;
};

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

