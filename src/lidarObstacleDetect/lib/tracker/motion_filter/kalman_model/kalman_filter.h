#pragma once

#include "common/registerer.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {


class IKalmanFilter {
public:
    IKalmanFilter() = default;
    virtual ~IKalmanFilter() = default;

    /**
     * @brief Filter init function
     * 
     * @param position object position
     * @param velocity object velocity, if not, default zero
     */
    virtual void Init(const Eigen::VectorXf& position, const Eigen::VectorXf& velocity) = 0;
    
    /**
     * @brief For state change, Global coordinate conversion
     * 
     * @param tf Attitude change of two adjacent frames
     */
    virtual void StateChange(const Eigen::Isometry3f& tf) = 0;

    /**
     * @brief For track point change
     * 
     * @param offset_x offset in X-Axis
     * @param offset_y offset in Y-Axis
     */
    virtual void StateChange(float offset_x, float offset_y) = 0;

    /**
     * @brief Prediction process
     * 
     * @param dt Predict duration
     * @return Eigen::VectorXf Predict result
     */
    virtual Eigen::VectorXf predict(const double& dt) = 0;

    /**
     * @brief Update process
     * 
     * @param position Measurement of position from sensor
     * @param velocity Measurement of velocity from sensor
     * @return Eigen::VectorXf Update result
     */
    virtual Eigen::VectorXf update(const Eigen::VectorXf& position, const Eigen::VectorXf& velocity) = 0;

    /**
     * @brief Get the State object
     * 
     * @return const Eigen::VectorXf& State and covariance
     */
    virtual const Eigen::VectorXf getState() const = 0;
    virtual const Eigen::MatrixXf getCovariance() const = 0; 
};

PERCEPTION_REGISTER_REGISTERER(IKalmanFilter);  // 注册基类
#define PERCEPTION_REGISTER_KALMAN_FILTER(name) \
  PERCEPTION_REGISTER_CLASS(IKalmanFilter, name) 

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

