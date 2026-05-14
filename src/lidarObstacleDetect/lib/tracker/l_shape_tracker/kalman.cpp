/**
 * @file kalman.cpp
 * @author xxx (xxx@depthink.tech)
 * @brief 
 * @version 0.1
 * @date 2021-04-25
 * 
 * @copyright Copyright (c) 2021 depthink
 * 
 */

#include "kalman.h"

KalmanFilter::KalmanFilter(double dt, const Eigen::MatrixXd& A,
                           const Eigen::MatrixXd& C, const Eigen::MatrixXd& Q,
                           const Eigen::MatrixXd& R, const Eigen::MatrixXd& P)
    : A(A),
      C(C),
      Q(Q),
      R(R),
      P0(P),
      m(C.rows()),
      n(A.rows()),
      t0(0),
      t(0),
      dt(dt),
      initialized(false),
      I(n, n),
      x_hat(n),
      x_hat_new(n) {
    I.setIdentity();
}

KalmanFilter::KalmanFilter() {}

KalmanFilter::KalmanFilter(const KalmanFilter& track) {
    this->A = track.A;
    this->C = track.C;
    this->Q = track.Q;
    this->R = track.R;
    this->P = track.P;
    this->K = track.K;
    this->P0 = track.P0;

    this->m = track.m;
    this->n = track.n;
    this->t0 = track.t0;
    this->t = track.t;
    this->dt = track.dt;
    this->initialized = track.initialized;
    this->I = track.I;
    this->x_hat = track.x_hat;
    this->x_hat_new = track.x_hat_new;
}

KalmanFilter& KalmanFilter::operator=(const KalmanFilter& track) {
    if (this == &track) {
        return *this;
    }

    this->A = track.A;
    this->C = track.C;
    this->Q = track.Q;
    this->R = track.R;
    this->P = track.P;
    this->K = track.K;
    this->P0 = track.P0;

    this->m = track.m;
    this->n = track.n;
    this->t0 = track.t0;
    this->t = track.t;
    this->dt = track.dt;
    this->initialized = track.initialized;
    this->I = track.I;
    this->x_hat = track.x_hat;
    this->x_hat_new = track.x_hat_new;
    return *this;
}

void KalmanFilter::init(double t0, const Eigen::VectorXd& x0) {
    x_hat = x0;
    P = P0;
    this->t0 = t0;
    t = t0;
    initialized = true;
}

void KalmanFilter::init() {
    x_hat.setZero();
    P = P0;
    t0 = 0;
    t = t0;
    initialized = true;
}

void KalmanFilter::predict() {
    if (!initialized) throw std::runtime_error("Filter is not initialized!");

    x_hat_new = A * x_hat;
    P = A * P * A.transpose() + Q;
    x_hat = x_hat_new;

    t += dt;
}

void KalmanFilter::predict(const double& dt) {
    this->dt = dt;

    if (!initialized) throw std::runtime_error("Filter is not initialized!");

    x_hat_new = A * x_hat;
    P = A * P * A.transpose() + Q;
    x_hat = x_hat_new;

    t += dt;
}

void KalmanFilter::update(const Eigen::VectorXd& y) {
    if (!initialized) throw std::runtime_error("Filter is not initialized!");

    K = P * C.transpose() * (C * P * C.transpose() + R).inverse();
    x_hat_new += K * (y - C * x_hat_new);
    P = (I - K * C) * P;
    x_hat = x_hat_new;
}

void KalmanFilter::changeStates(const Eigen::VectorXd& new_states) {
    if (!initialized) throw std::runtime_error("Filter is not initialized!");
    if (x_hat.size() != new_states.size())
        throw std::runtime_error("State vectors do not have the same size");
    x_hat = new_states;
}

void KalmanFilter::update(const Eigen::VectorXd& y, double dt,
                          const Eigen::MatrixXd A) {
    this->A = A;
    this->dt = dt;
    predict();
    update(y);
}

void KalmanFilter::update(const Eigen::VectorXd& y, double dt) {
    this->dt = dt;
    predict();
    update(y);
}

