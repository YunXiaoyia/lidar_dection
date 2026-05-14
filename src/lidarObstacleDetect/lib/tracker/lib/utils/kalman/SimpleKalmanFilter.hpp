#pragma once

#include <Eigen/Core>
#include "kinematic_model/cv_model.hpp"
#include "kinematic_model/shape_filter_model.hpp"
#include "../../../motion_filter/kalman_model/kalman_filter.h"

namespace Kalman {

// Simplified non-template Kalman filter for CV state
class SimpleCVKalmanFilter : public highway::perception::track::IKalmanFilter {
private:
    Kalman::cv::State x_;        // State vector
    Eigen::MatrixXf P_;          // State covariance matrix

public:
    SimpleCVKalmanFilter() {
        P_.resize(4, 4);
        P_.setIdentity();
        x_.setZero();
    }

    // IKalmanFilter interface implementation
    void Init(const Eigen::VectorXf& position, const Eigen::VectorXf& velocity) override {
        x_.resize(4);
        x_.setZero();
        if (position.size() >= 2) {
            x_(0) = position(0);
            x_(1) = position(1);
        }
        if (velocity.size() >= 2) {
            x_(2) = velocity(0);
            x_(3) = velocity(1);
        }
        P_.resize(4, 4);
        P_.setIdentity();
    }
    
    void StateChange(const Eigen::Isometry3f& tf) override {
        Eigen::Vector3f pos = x_.head<3>();
        pos = tf * pos;
        x_.head<3>() = pos;
    }

    void StateChange(float offset_x, float offset_y) override {
        x_(0) += offset_x;
        x_(1) += offset_y;
    }

    Eigen::VectorXf predict(const double& dt) override {
        // Simple constant velocity prediction
        Eigen::VectorXf x_pred = x_;
        x_pred(0) += x_(2) * dt;  // x += vx * dt
        x_pred(1) += x_(3) * dt;  // y += vy * dt
        
        x_ = x_pred;
        return x_;
    }

    Eigen::VectorXf update(const Eigen::VectorXf& position, const Eigen::VectorXf& velocity) override {
        // Simple update with measurement
        float alpha = 0.3f;  // Learning rate
        if (position.size() >= 2) {
            x_(0) = (1.0f - alpha) * x_(0) + alpha * position(0);  // Update x
            x_(1) = (1.0f - alpha) * x_(1) + alpha * position(1);  // Update y
        }
        if (velocity.size() >= 2) {
            x_(2) = (1.0f - alpha) * x_(2) + alpha * velocity(0);  // Update vx
            x_(3) = (1.0f - alpha) * x_(3) + alpha * velocity(1);  // Update vy
        }
        
        return x_;
    }

    const Eigen::VectorXf getState() const override { 
        return x_; 
    }
    
    const Eigen::MatrixXf getCovariance() const override { 
        return P_; 
    }

    // Legacy methods for compatibility
    void init(const Kalman::cv::State& x) {
        x_ = x;
        P_.resize(x.size(), x.size());
        P_.setIdentity();
    }

    const Kalman::cv::State& getStateCV() const { return x_; }

    void setCovariance(const Eigen::MatrixXf& P) { 
        if (P.rows() == P_.rows() && P.cols() == P_.cols()) {
            P_ = P; 
        }
    }

    // Predict step (legacy)
    Kalman::cv::State predict(const Kalman::cv::SystemModel& system_model, const Kalman::cv::Control& u) {
        // Simple constant velocity prediction
        float dt = u.dt();
        Kalman::cv::State x_pred = x_;
        x_pred(0) += x_(2) * dt;  // x += vx * dt
        x_pred(1) += x_(3) * dt;  // y += vy * dt
        
        x_ = x_pred;
        return x_;
    }

    // Update step (legacy)
    Kalman::cv::State update(const Kalman::cv::PositionModel& measurement_model, const Kalman::cv::PositionMeasurement& z) {
        // Simple update with measurement
        float alpha = 0.3f;  // Learning rate
        x_(0) = (1.0f - alpha) * x_(0) + alpha * z.x();  // Update x
        x_(1) = (1.0f - alpha) * x_(1) + alpha * z.y();  // Update y
        
        return x_;
    }
};

// Simplified non-template Kalman filter for Shape state
class SimpleShapeKalmanFilter {
private:
    Kalman::shape::State x_;     // State vector
    Eigen::MatrixXf P_;          // State covariance matrix

public:
    SimpleShapeKalmanFilter() {
        P_.resize(3, 3);
        P_.setIdentity();
        x_.setZero();
    }

    // Initialize filter with state
    void init(const Kalman::shape::State& x) {
        x_ = x;
        P_.resize(x.size(), x.size());
        P_.setIdentity();
    }

    // Get current state
    const Kalman::shape::State& getState() const { return x_; }

    // Get current covariance
    const Eigen::MatrixXf& getCovariance() const { return P_; }

    // Set covariance
    void setCovariance(const Eigen::MatrixXf& P) { 
        if (P.rows() == P_.rows() && P.cols() == P_.cols()) {
            P_ = P; 
        }
    }

    // Predict step
    Kalman::shape::State predict(const Kalman::shape::SystemModel& system_model, const Kalman::shape::Control& u) {
        // For shape, prediction is mostly identity (shapes don't change much)
        return x_;
    }

    // Update step
    Kalman::shape::State update(const Kalman::shape::MeasurementModel& measurement_model, const Kalman::shape::ShapeMeasurement& z) {
        // Simple update with measurement
        float alpha = 0.3f;  // Learning rate
        x_(0) = (1.0f - alpha) * x_(0) + alpha * z.l1();   // Update l1
        x_(1) = (1.0f - alpha) * x_(1) + alpha * z.l2();   // Update l2
        x_(2) = (1.0f - alpha) * x_(2) + alpha * z.theta(); // Update theta
        
        return x_;
    }
};

} // namespace Kalman
