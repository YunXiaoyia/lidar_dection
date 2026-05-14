#pragma once

#include <Eigen/Core>
#include <Eigen/LU>

namespace Kalman {

template<typename StateType>
class ExtendedKalmanFilter {
private:
    StateType x_;        // State vector
    Eigen::MatrixXf P_;  // State covariance matrix

public:
    ExtendedKalmanFilter() = default;

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
        // In a real implementation, this would use the system model
        StateType x_pred = x_;
        
        // Simple constant velocity prediction
        float dt = u.dt();
        if (x_.size() >= 4) {
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
        // In a real implementation, this would use the measurement model
        
        if (x_.size() >= 4) {
            // Simple update with measurement
            float alpha = 0.3f;  // Learning rate
            x_(0) = (1.0f - alpha) * x_(0) + alpha * z.x();  // Update x
            x_(1) = (1.0f - alpha) * x_(1) + alpha * z.y();  // Update y
        }
        
        return x_;
    }
};

} // namespace Kalman
