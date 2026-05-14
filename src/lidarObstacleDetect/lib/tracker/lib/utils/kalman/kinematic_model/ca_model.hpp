#pragma once

#include <Eigen/Core>

namespace Kalman {
namespace ca {

// State vector for Constant Acceleration model: [x, y, vx, vy, ax, ay]
using State = Eigen::Matrix<float, 6, 1>;

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

// System model for Constant Acceleration
class SystemModel {
private:
    Eigen::Matrix<float, 6, 6> A_;  // State transition matrix
    Eigen::Matrix<float, 6, 6> Q_;  // Process noise covariance

public:
    SystemModel() {
        A_.setIdentity();
        Q_.setIdentity();
    }

    const Eigen::Matrix<float, 6, 6>& getCovariance() const { return Q_; }
    void setCovariance(const Eigen::Matrix<float, 6, 6>& Q) { Q_ = Q; }

    const Eigen::Matrix<float, 6, 6>& getA() const { return A_; }
    void setA(const Eigen::Matrix<float, 6, 6>& A) { A_ = A; }

    // Update state transition matrix for given dt
    void updateA(float dt) {
        float dt2 = dt * dt;
        float dt3 = dt2 * dt / 2.0f;
        
        A_ << 1, 0, dt, 0, dt3, 0,
              0, 1, 0, dt, 0, dt3,
              0, 0, 1, 0, dt, 0,
              0, 0, 0, 1, 0, dt,
              0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 1;
    }
};

// Measurement model for position
class PositionModel {
private:
    Eigen::Matrix<float, 2, 6> H_;  // Measurement matrix
    Eigen::Matrix2f R_;             // Measurement noise covariance

public:
    PositionModel() {
        H_ << 1, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0;
        R_.setIdentity();
    }

    const Eigen::Matrix2f& getCovariance() const { return R_; }
    void setCovariance(const Eigen::Matrix2f& R) { R_ = R; }

    const Eigen::Matrix<float, 2, 6>& getH() const { return H_; }
    void setH(const Eigen::Matrix<float, 2, 6>& H) { H_ = H; }
};

} // namespace ca
} // namespace Kalman
