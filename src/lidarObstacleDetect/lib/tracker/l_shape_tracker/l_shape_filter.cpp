/**
 * @file l_shape_tracker.cpp
 * @author xxx (xxx@depthink.tech) xxx (xxx@depthink.tech)
 * @brief
 * @version 0.1
 * @date 2021-04-25
 *
 * @copyright Copyright (c) 2021 depthink
 *
 */

#include "l_shape_filter.h"

#include "kalman.h"
#include <common/lidar_perception_log.h>
#include <string>

#include "common/Macros.h"
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_BEGIN
namespace track {

double normalize_angle_positive(double angle) {
    // Normalizes the angle to be 0 to 2*M_PI.
    // It takes and returns radians.
    return fmod(fmod(angle, 2.0 * M_PI) + 2.0 * M_PI, 2.0 * M_PI);
}

// normalize_angle [-pi, pi]
double normalize_angle(double angle) {
    // Normalizes the angle to be -M_PI circle to +M_PI circle
    // It takes and returns radians.
    double a = normalize_angle_positive(angle);
    if (a > M_PI) a -= 2.0 * M_PI;
    return a;
}

double shortest_angular_distance(double from, double to) {
    // Given 2 angles, this returns the shortest angular difference.
    // The inputs and outputs are radians.
    // The result would always be -pi <= result <= pi.
    // Adding the result to "from" results to "to".
    return normalize_angle(to - from);
}

LshapeFilter::LshapeFilter() {}

LshapeFilter::LshapeFilter(const LshapeFilter& track) {
    this->thetaL1_old = track.thetaL1_old;
    this->switchCornerAngle = track.switchCornerAngle;
    this->shape_kf = track.shape_kf;
    this->dynamic_kf = track.dynamic_kf;
}

LshapeFilter& LshapeFilter::operator=(const LshapeFilter& track) {
    if (this == &track) {
        return *this;
    }

    this->thetaL1_old = track.thetaL1_old;
    this->switchCornerAngle = track.switchCornerAngle;
    this->shape_kf = track.shape_kf;
    this->dynamic_kf = track.dynamic_kf;
    this->last_tracked_time_ = track.last_tracked_time_;
    return *this;
}

LshapeFilter::LshapeFilter(const double& x_corner, const double& y_corner, const double& L1, const double& L2,
                             const double& theta, const double& cur_time) {
    
    last_tracked_time_ = cur_time;

    double dt = 0.09999;
    // Initialization of Dynamic Kalman Filter
    int n = 4;         // Number of states
    int m = 2;         // Number of measurements
    MatrixXd A(n, n);  // System dynamics matrix
    MatrixXd C(m, n);  // Output matrix
    MatrixXd Q(n, n);  // Process noise covariance
    MatrixXd R(m, m);  // Measurement noise covariance
    MatrixXd P(n, n);  // Estimate error covariance

    A << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1;

    C << 1, 0, 0, 0, 0, 1, 0, 0;

    // 从调试结果看:
    // Q调大收敛速度快，但稳定性下降；Q调小收敛速度慢，但稳定性好
    Q << 1, 0, 0, 0, 
        0, 1, 0, 0, 
        0, 0, 5, 0, 
        0, 0, 0, 5;

    // 从调试结果看:
    // 影响收敛到真值附近的幅度（高低），R越大，越高；R越小，越低。
    R.setIdentity();
    R *= 20;

    // P.setIdentity();
    // 从调试结果看: 影响初始值收敛到真值的速度。
    P << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1650, 0, 0, 0, 0, 1650;

    KalmanFilter dynamic_kalman_filter(dt, A, C, Q, R, P);
    this->dynamic_kf = dynamic_kalman_filter;

    VectorXd x0_dynamic(n);
    x0_dynamic << x_corner, y_corner, 0, 0;
    dynamic_kf.init(0, x0_dynamic);  // kalamn初始化的时间为0

    // Initialization of Shape Kalman Filter
    n = 4;              // Number of states
    m = 3;              // Number of measurements
    MatrixXd As(n, n);  // System dynamics matrix
    MatrixXd Cs(m, n);  // Output matrix
    MatrixXd Qs(n, n);  // Process noise covariance
    MatrixXd Rs(m, m);  // Measurement noise covariance
    MatrixXd Ps(n, n);  // Estimate error covariance

    // 状态向量为 l1,l2,theta，(第四个未知 不知道是啥)  与时间有关系的只有角度
    As << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, dt, 0, 0, 0, 1;

    // 观测到预测的映射矩阵，观测为三个变量，预测为四个
    Cs << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0;

    // Qs << 1e-4, 0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 1e-3, 0, 0, 0, 0, 1e-3;

    Qs << 1e-3, 0, 0, 0, 0, 1e-3, 0, 0, 0, 0, 0.001, 0, 0, 0, 0, 0.01;

    Ps.setIdentity();
    // Ps << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10000, 0, 0, 0, 0, 10000;

    KalmanFilter shape_kalman_filter(dt, As, Cs, Qs, Rs, Ps);
    this->shape_kf = shape_kalman_filter;

    VectorXd x0_shape(n);
    double L1init = 0.1;
    double L2init = 0.1;
    if (L1 > L1init) {
        L1init = L1;
    }
    if (L2 > L2init) {
        L2init = L2;
    }
    x0_shape << L1init, L2init, theta, 0;
    shape_kf.init(0, x0_shape);  // kalamn初始化的时间为0

    thetaL1_old = theta;
    switchCornerAngle = 1.0;
}

void LshapeFilter::predict(const double& cur_time) {

    // double dt = cur_time - last_tracked_time_;
    // last_tracked_time_ = cur_time;

    // dynamic_kf.A(0, 2) = dt;
    // dynamic_kf.A(1, 3) = dt;
    // dynamic_kf.predict(dt);

    // shape_kf.A(2, 3) = dt;
    // shape_kf.predict(dt);
}

void LshapeFilter::update(const double& x_corner, const double& y_corner, const double& L1, const double& L2,
                           const double& thetaL1, const double& cur_time) {
    
    double dt = cur_time - last_tracked_time_;
    last_tracked_time_ = cur_time;

    // state predict
    dynamic_kf.A(0, 2) = dt;
    dynamic_kf.A(1, 3) = dt;
    dynamic_kf.predict(dt);

    shape_kf.A(2, 3) = dt;
    shape_kf.predict(dt);

    // 判断角点是否发生转换
    int is_switch = detectCornerPointSwitch(thetaL1_old, thetaL1);

    // 若角点发生转换则转换模型中角点的位置
    if (is_switch != 0) {
        Vector4d new_dynamic_states = dynamic_kf.state();
        new_dynamic_states(0) = x_corner;
        new_dynamic_states(1) = y_corner;
        dynamic_kf.changeStates(new_dynamic_states);
        history_dynamic_meas_ = new_dynamic_states.head(2);
    }

    double norm = normalize_angle(shape_kf.state()(2));
    double distance = shortest_angular_distance(norm, thetaL1);
    double theta = distance + shape_kf.state()(2);
    thetaL1_old = thetaL1;

    // Update Dynamic Kalman Filter
    if (is_switch == 0) {
        Vector2d y;
        y << x_corner, y_corner;
        dynamic_kf.update(y);

        history_dynamic_meas_ = y;
    }

    // Update Shape Kalman Filter
    shape_kf.R << pow(L1, -1.0), 0, 0, 0, pow(L2, -1.0), 0, 0, 0, 1;
    Vector3d shape_measurements;
    shape_measurements << L1, L2, theta;

    shape_kf.predict(dt);
    shape_kf.update(shape_measurements);
    history_shape_meas_ = shape_measurements;
}

void LshapeFilter::transformToCurrent(const Eigen::Isometry3d& tf) {
    Vector4d dynamic_states = dynamic_kf.state();

    Vector3d reference_point;
    reference_point << dynamic_states(0), dynamic_states(1), 0.0;
    reference_point = tf * reference_point;
    dynamic_states.head(2) = reference_point.head(2);

    Vector3d velocity;
    velocity << dynamic_states(2), dynamic_states(3), 0.0;
    velocity = tf.rotation() * velocity;
    dynamic_states.tail(2) = velocity.head(2);

    dynamic_kf.changeStates(dynamic_states);
}

Eigen::VectorXf LshapeFilter::getMotionState(){
    return dynamic_kf.state().cast<float>();
}

void LshapeFilter::BoxModel(Eigen::Vector3f& center, Eigen::Vector3f& velocity,
                             std::array<Eigen::Vector3f, 4>& corner_point_array, Eigen::Vector3f& extent, float& direction) {
    // Eigen::Vector3d shape_filter_states =  shape_kf.state();
    // Eigen::Vector2d dynamic_filter_states = dynamic_kf.state();

    Eigen::Vector3d shape_filter_states = history_shape_meas_;      // shape: L1, L2, theta
    Eigen::Vector2d dynamic_filter_states = history_dynamic_meas_;  // 角点 x, y

    double L1 = shape_filter_states(0);
    double L2 = shape_filter_states(1);
    double theta = shape_filter_states(2);

    // Equations 30 of "L-Shape Model Switching-Based precise motion tracking of
    // moving vehicles"
    double ex = (L1 * cos(theta) + L2 * sin(theta)) / 2;
    double ey = (L1 * sin(theta) - L2 * cos(theta)) / 2;

    center(0) = dynamic_filter_states(0) + ex;
    center(1) = dynamic_filter_states(1) + ey;
    center(2) = 0.0f;
    // Equations 31 of "L-Shape Model Switching-Based precise motion tracking of
    // moving vehicles"
    velocity(0) = dynamic_kf.state()(2);
    velocity(1) = dynamic_kf.state()(3);
    velocity(2) = 0.0f;

    Eigen::Vector3f corner_point;
    corner_point(0) = dynamic_filter_states(0);
    corner_point(1) = dynamic_filter_states(1);
    corner_point(2) = 0.0f;
    corner_point_array[0] = corner_point;

    corner_point(0) += L1 * cos(theta);
    corner_point(1) += L1 * sin(theta);
    corner_point(2) = 0.0f;
    corner_point_array[1] = corner_point;

    corner_point(0) += L2 * sin(theta);
    corner_point(1) -= L2 * cos(theta);
    corner_point(2) = 0.0f;
    corner_point_array[2] = corner_point;

    corner_point(0) -= L1 * cos(theta);
    corner_point(1) -= L1 * sin(theta);
    corner_point(2) = 0.0f;
    corner_point_array[3] = corner_point;

    findOrientation(direction, extent(0), extent(1));
}



void LshapeFilter::findOrientation(float& direction, float& length, float& width){
    //This function finds the orientation of a moving object, when given an L-shape orientation
    std::vector<float> angles;
    // double angle_norm = normalize_angle(shape_kf.state()(2));
    double angle_norm = normalize_angle(history_shape_meas_[2]);
    angles.push_back(angle_norm);
    angles.push_back(angle_norm + M_PI);
    angles.push_back(angle_norm + M_PI / 2);
    angles.push_back(angle_norm + 3 * M_PI / 2);

    double vsp = atan2(dynamic_kf.state()(3), dynamic_kf.state()(2));
    double min = 1.56;
    double distance = 0.0, orientation = 0.0;
    int pos = -1;
    for(int i = 0; i < 4; ++i){
        distance = abs(shortest_angular_distance(vsp, angles[i]));
        if(distance < min){
            min = distance;
            orientation = normalize_angle(angles[i]);
            pos = i;
        }
    }
    if(pos == 0 || pos == 1){
        // length = shape_kf.state()(0);
        // width = shape_kf.state()(1);
        length = history_shape_meas_[0];
        width = history_shape_meas_[1];
    }
    else{
        // length = shape_kf.state()(1);
        // width = shape_kf.state()(0);
        length = history_shape_meas_[1];
        width = history_shape_meas_[0];
    }
    direction = normalize_angle(orientation);
}



double LshapeFilter::findTurn(const double& new_angle, const double& old_angle) {
    // https://math.stackexchange.com/questions/1366869/calculating-rotation-direction-between-two-angles
    double theta_pro = new_angle - old_angle;
    double turn = 0;
    if (-M_PI <= theta_pro && theta_pro <= M_PI) {
        turn = theta_pro;
    } else if (theta_pro > M_PI) {
        turn = theta_pro - 2 * M_PI;
    } else if (theta_pro < -M_PI) {
        turn = theta_pro + 2 * M_PI;
    }
    return turn;
}

int LshapeFilter::detectCornerPointSwitch(const double& from, const double& to) {
    // Corner Point Switch Detection
    double turn = findTurn(from, to);
    if (turn < -1.0 * switchCornerAngle) {
        this->CounterClockwisePointSwitch();
        return -1;
    } else if (turn > switchCornerAngle) {
        this->ClockwisePointSwitch();
        return 1;
    }

    return 0;
}

/// Switch different model
void LshapeFilter::ClockwisePointSwitch() {
    Vector4d new_shape_states = shape_kf.state();
    double L1 = shape_kf.state()(0);
    double L2 = shape_kf.state()(1);

    // L1 和 L2 交换
    new_shape_states(0) = L2;
    new_shape_states(1) = L1;
    new_shape_states(2) = shape_kf.state()(2) - M_PI_2;
    shape_kf.changeStates(new_shape_states);
}

/// Switch different model
void LshapeFilter::CounterClockwisePointSwitch() {
    Vector4d new_shape_states = shape_kf.state();
    double L1 = shape_kf.state()(0);
    double L2 = shape_kf.state()(1);

    // L1 和 L2 交换
    new_shape_states(0) = L2;
    new_shape_states(1) = L1;
    new_shape_states(2) = shape_kf.state()(2) + M_PI_2;
    shape_kf.changeStates(new_shape_states);
}

}  // namespace track
HIGHWAY_PERCEPTION_LIBS_NAMESPACE_END

