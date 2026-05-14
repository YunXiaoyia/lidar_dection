#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <type_traits>
#include "kinematic_model/cv_model.hpp"
#include "kinematic_model/shape_filter_model.hpp"

namespace Kalman {

template<typename StateType>
class NormalKalmanFilter {
private:
    StateType x_;        // State vector
    Eigen::MatrixXf P_;  // State covariance matrix

public:
    NormalKalmanFilter() = default;

    // Initialize filter with state
    void init(const StateType& x) {
        x_ = x;
        P_.resize(x.size(), x.size());
        P_.setIdentity();
    }

    // Get current state
    const StateType& getState() const { return x_; }

    // Get current covariance
    const Eigen::MatrixXf& getCovariance() const { return P_; }

    // Set covariance
    void setCovariance(const Eigen::MatrixXf& P) { 
        if (P.rows() == P_.rows() && P.cols() == P_.cols()) {
            P_ = P; 
        }
    }

    // Predict step (simplified version)
    template<typename SystemModel, typename ControlType>
    StateType predict(const SystemModel& system_model, const ControlType& u) {
        // For simplicity, just apply basic prediction
        StateType x_pred = x_;
        
        // Simple constant velocity prediction
        if (x_.size() >= 4) {
            float dt = u.dt();  // Use actual dt from control
            x_pred(0) += x_(2) * dt;  // x += vx * dt
            x_pred(1) += x_(3) * dt;  // y += vy * dt
        }
        
        x_ = x_pred;
        return x_;
    }

    // Update step (simplified version)
    template<typename MeasurementModel, typename MeasurementType>
    StateType update(const MeasurementModel& measurement_model, const MeasurementType& z) {
        // For simplicity, just apply basic update
        if (x_.size() >= 4) {
            // Simple update with measurement
            float alpha = 0.3f;  // Learning rate
            if constexpr (std::is_same_v<MeasurementType, Kalman::cv::PositionMeasurement>) {
                x_(0) = (1.0f - alpha) * x_(0) + alpha * z.x();  // Update x
                x_(1) = (1.0f - alpha) * x_(1) + alpha * z.y();  // Update y
            } else if constexpr (std::is_same_v<MeasurementType, Kalman::shape::ShapeMeasurement>) {
                // For shape filter, update all three states
                if (x_.size() >= 3) {
                    x_(0) = (1.0f - alpha) * x_(0) + alpha * z.l1();  // Update l1
                    x_(1) = (1.0f - alpha) * x_(1) + alpha * z.l2();  // Update l2
                    x_(2) = (1.0f - alpha) * x_(2) + alpha * z.theta();  // Update theta
                }
            }
        }
        
        return x_;
    }
};

} // namespace Kalman
