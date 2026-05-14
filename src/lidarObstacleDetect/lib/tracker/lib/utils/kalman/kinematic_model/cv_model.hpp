#pragma once

#include <Eigen/Core>

namespace Kalman {
namespace cv {

// State vector for Constant Velocity model: [x, y, vx, vy]
using State = Eigen::Vector4f;

// Control input: dt
struct Control {
    float dt_;
    float& dt() { return dt_; }
    const float& dt() const { return dt_; }
};

// Position measurement: [x, y]
struct PositionMeasurement {
    float x_, y_;
    float& x() { return x_; }
    float& y() { return y_; }
    const float& x() const { return x_; }
    const float& y() const { return y_; }
};

// System model for Constant Velocity
class SystemModel {
private:
    Eigen::Matrix4f A_;  // State transition matrix
    Eigen::Matrix4f Q_;  // Process noise covariance

public:
    SystemModel() {
        A_.setIdentity();
        Q_.setIdentity();
    }

    const Eigen::Matrix4f& getCovariance() const { return Q_; }
    void setCovariance(const Eigen::Matrix4f& Q) { Q_ = Q; }

    const Eigen::Matrix4f& getA() const { return A_; }
    void setA(const Eigen::Matrix4f& A) { A_ = A; }

    // Update state transition matrix for given dt
    void updateA(float dt) {
        A_ << 1, 0, dt, 0,
              0, 1, 0, dt,
              0, 0, 1, 0,
              0, 0, 0, 1;
    }
};

// Measurement model for position
class PositionModel {
private:
    Eigen::Matrix<float, 2, 4> H_;  // Measurement matrix
    Eigen::Matrix2f R_;             // Measurement noise covariance

public:
    PositionModel() {
        H_ << 1, 0, 0, 0,
              0, 1, 0, 0;
        R_.setIdentity();
    }

    const Eigen::Matrix2f& getCovariance() const { return R_; }
    void setCovariance(const Eigen::Matrix2f& R) { R_ = R; }

    const Eigen::Matrix<float, 2, 4>& getH() const { return H_; }
    void setH(const Eigen::Matrix<float, 2, 4>& H) { H_ = H; }
};

} // namespace cv
} // namespace Kalman
