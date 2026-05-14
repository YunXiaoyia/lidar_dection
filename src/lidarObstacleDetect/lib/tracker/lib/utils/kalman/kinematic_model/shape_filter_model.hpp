#pragma once

#include <Eigen/Core>

namespace Kalman {
namespace shape {

// State vector for shape filter: [l1, l2, theta] 
using State = Eigen::Vector3f;

// Control input for shape filter
struct Control {
    float dt_;
    float& dt() { return dt_; }
    const float& dt() const { return dt_; }
};

// Shape measurement: [l1, l2, theta]
struct ShapeMeasurement {
    float l1_, l2_, theta_;
    float& l1() { return l1_; }
    float& l2() { return l2_; }
    float& theta() { return theta_; }
    const float& l1() const { return l1_; }
    const float& l2() const { return l2_; }
    const float& theta() const { return theta_; }
};

// System model for shape (identity with small process noise)
class SystemModel {
private:
    Eigen::Matrix3f A_;  // State transition matrix (identity)
    Eigen::Matrix3f Q_;  // Process noise covariance

public:
    SystemModel() {
        A_.setIdentity();
        Q_.setIdentity();
        Q_ *= 0.01f;  // Small process noise
    }

    const Eigen::Matrix3f& getCovariance() const { return Q_; }
    void setCovariance(const Eigen::Matrix3f& Q) { Q_ = Q; }

    const Eigen::Matrix3f& getA() const { return A_; }
    void setA(const Eigen::Matrix3f& A) { A_ = A; }
};

// Measurement model for shape
class MeasurementModel {
private:
    Eigen::Matrix3f H_;  // Measurement matrix (identity)
    Eigen::Matrix3f R_;  // Measurement noise covariance

public:
    MeasurementModel() {
        H_.setIdentity();
        R_.setIdentity();
        R_ *= 0.1f;  // Measurement noise
    }

    const Eigen::Matrix3f& getCovariance() const { return R_; }
    void setCovariance(const Eigen::Matrix3f& R) { R_ = R; }

    const Eigen::Matrix3f& getH() const { return H_; }
    void setH(const Eigen::Matrix3f& H) { H_ = H; }
};

} // namespace shape
} // namespace Kalman
