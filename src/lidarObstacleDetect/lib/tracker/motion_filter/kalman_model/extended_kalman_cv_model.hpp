#pragma once

#include "lib/utils/kalman/kinematic_model/cv_model.hpp"
#include "lib/utils/kalman/ExtendedKalmanFilter.hpp"

#include "motion_filter/kalman_model/kalman_filter.h"

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {


class ExtendedKalmanCV : public IKalmanFilter {
public:
    ExtendedKalmanCV() = default;
    virtual ~ExtendedKalmanCV() = default;

    void Init(const Eigen::VectorXf& position, const Eigen::VectorXf& velocity) {
        
        // init system covariance
        auto system_covariance = cv_sys_model_.getCovariance();
        system_covariance.setZero();
        system_covariance(0, 0) = 1;
        system_covariance(1, 1) = 1;
        system_covariance(2, 2) = 3;
        system_covariance(3, 3) = 0.5;
        cv_sys_model_.setCovariance(system_covariance);

        // init measurement covariance
        auto meas_covariance = cv_meas_model_.getCovariance();
        meas_covariance.setZero();
        meas_covariance(0, 0) = 0.5;
        meas_covariance(1, 1) = 3;
        cv_meas_model_.setCovariance(meas_covariance);

        // init system state
        Kalman::cv::State init_state;
        init_state(0) = position(0);
        init_state(1) = position(1);
        init_state(2) = velocity(0);
        init_state(3) = velocity(1);
        ekf_.init(init_state);
    }

    void StateChange(const Eigen::Isometry3f& tf) {
        auto global_states = ekf_.getState();

        // Position
        Eigen::Vector3f pos_point;
        pos_point << global_states(0), global_states(1), 0.0;
        pos_point = (tf * pos_point).eval();
        global_states.head(2) = pos_point.head(2);

        // Velocity
        Eigen::Vector3f velocity;
        velocity << global_states(2), global_states(3), 0.0;
        velocity = (tf.rotation() * velocity).eval();
        global_states[2] = velocity[0];
        global_states[3] = velocity[1];

        ekf_.init(global_states);
    }

    void StateChange(float offset_x, float offset_y) {
        auto global_states = ekf_.getState();

        global_states[0] = global_states[0] + offset_x;
        global_states[1] = global_states[1] + offset_y;

        ekf_.init(global_states);
    }

    Eigen::VectorXf predict(const double& dt) {

        Kalman::cv::Control u;
        u.dt() = dt;

        return ekf_.predict(cv_sys_model_, u).block<4, 1>(0, 0);
    }

    Eigen::VectorXf update(const Eigen::VectorXf& position, const Eigen::VectorXf& velocity) {

        Kalman::cv::PositionMeasurement pos_meas;
        pos_meas.x() = position(0);
        pos_meas.y() = position(1);
        
        return ekf_.update(cv_meas_model_, pos_meas).block<4, 1>(0, 0);
    }

private:
    Kalman::cv::SystemModel cv_sys_model_;
    Kalman::cv::PositionModel cv_meas_model_;
    Kalman::ExtendedKalmanFilter<Kalman::cv::State> ekf_;

public:
    const Eigen::VectorXf getState() const { return ekf_.getState();};
    const Eigen::MatrixXf getCovariance() const { return ekf_.getCovariance().block<4, 4>(0, 0);};
};
PERCEPTION_REGISTER_KALMAN_FILTER(ExtendedKalmanCV);

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

