#pragma once

#include "motion_filter/kalman_model/kalman_filter.h"
#include "lib/utils/kalman/ExtendedKalmanFilter.hpp"
#include "lib/utils/kalman/NormalKalmanFilter.hpp"
#include "lib/utils/kalman/kinematic_model/cv_model.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

class ExtendedKalmanFilterImpl : public IKalmanFilter {
public:
    ExtendedKalmanFilterImpl() = default;
    ~ExtendedKalmanFilterImpl() = default;

    void Init(const Eigen::VectorXf& position, const Eigen::VectorXf& velocity) override {
        // 使用 CV 模型，状态向量 [x, y, vx, vy]
        state_.resize(4);
        state_ << position(0), position(1), velocity(0), velocity(1);
        
        kalman_filter_.init(state_);
        
        // 设置初始协方差
        Eigen::Matrix4f P = Eigen::Matrix4f::Identity() * 1.0f;
        kalman_filter_.setCovariance(P);
    }
    
    void StateChange(const Eigen::Isometry3f& tf) override {
        // 简化实现，只应用平移
        Eigen::Vector3f translation = tf.translation();
        state_(0) += translation.x();
        state_(1) += translation.y();
    }

    void StateChange(float offset_x, float offset_y) override {
        state_(0) += offset_x;
        state_(1) += offset_y;
    }

    Eigen::VectorXf predict(const double& dt) override {
        // 简化的预测实现
        StateType pred_state = state_;
        pred_state(0) += state_(2) * dt;  // x += vx * dt
        pred_state(1) += state_(3) * dt;  // y += vy * dt
        return pred_state;
    }

    Eigen::VectorXf update(const Eigen::VectorXf& position, const Eigen::VectorXf& velocity) override {
        // 简化的更新实现
        float alpha = 0.3f;  // 学习率
        state_(0) = (1.0f - alpha) * state_(0) + alpha * position(0);  // Update x
        state_(1) = (1.0f - alpha) * state_(1) + alpha * position(1);  // Update y
        state_(2) = (1.0f - alpha) * state_(2) + alpha * velocity(0);  // Update vx
        state_(3) = (1.0f - alpha) * state_(3) + alpha * velocity(1);  // Update vy
        
        return state_;
    }

    const Eigen::VectorXf getState() const override {
        return state_;
    }

    const Eigen::MatrixXf getCovariance() const override {
        return kalman_filter_.getCovariance();
    }

private:
    using StateType = Eigen::Matrix<float, 4, 1>;
    Kalman::ExtendedKalmanFilter<StateType> kalman_filter_;
    StateType state_;
};

PERCEPTION_REGISTER_KALMAN_FILTER(ExtendedKalmanFilterImpl);

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END
