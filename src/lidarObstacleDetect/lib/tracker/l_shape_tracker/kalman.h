/**
 * @file kalman.h
 * @author xxx (xxx@depthink.tech)
 * @brief
 * @version 0.1
 * @date 2021-04-25
 *
 * @copyright Copyright (c) 2021 depthink
 *
 */

#ifndef KALMAN_H
#define KALMAN_H

#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include "Eigen/src/Core/Matrix.h"

/// @brief KalmanFilter
class KalmanFilter {
   public:
    /**
     * Create a Kalman filter with the specified matrices.
     *   A - System dynamics matrix
     *   C - Output matrix
     *   Q - Process noise covariance
     *   R - Measurement noise covariance
     *   P - Estimate error covariance
     */
    /// Matrices for computation
    Eigen::MatrixXd A, C, Q, R, P, K, P0;

    /**
     * @brief Construct a new Kalman Filter object
     * 
     * @param dt 
     * @param A 
     * @param C 
     * @param Q 
     * @param R 
     * @param P 
     */
    KalmanFilter(double dt, const Eigen::MatrixXd& A, const Eigen::MatrixXd& C, const Eigen::MatrixXd& Q,
                 const Eigen::MatrixXd& R, const Eigen::MatrixXd& P);

    /**
     * Create a blank estimator.
     */
    KalmanFilter();
    KalmanFilter(const KalmanFilter&);
    KalmanFilter& operator=(const KalmanFilter&);

    /**
     * Initialize the filter with initial states as zero.
     */
    void init();

    /**
     * Initialize the filter with a guess for initial states.
     */
    void init(double t0, const Eigen::VectorXd& x0);

    /**
     * Update the estimated state based on measured values. The
     * time step is assumed to remain constant.
     */
    void predict();

    /**
     * @brief 
     * 
     * @param dt 
     */
    void predict(const double& dt);

    /**
     * @brief 
     * 
     * @param y 
     */
    void update(const Eigen::VectorXd& y);

    /**
     * @brief 
     * 
     * @param y 
     * @param dt 
     */
    void update(const Eigen::VectorXd& y, double dt);

    /**
     * Update the estimated state based on measured values,
     * using the given time step and dynamics matrix.
     */
    void update(const Eigen::VectorXd& y, double dt, const Eigen::MatrixXd A);

    /**
     * @brief 
     * 
     * @param new_states 
     */
    void changeStates(const Eigen::VectorXd& new_states);

    /**
     * Return the current state and time.
     */
    Eigen::VectorXd state() { return x_hat; };

    Eigen::MatrixXd stateCovariance() { return P; };

    /// @brief
    double time() { return t; };

   private:
    /// System dimensions
    int m, n;

    /// Initial and current time
    double t0, t;

    /// Discrete time step
    double dt;

    /// Is the filter initialized?
    bool initialized;

    /// n-size identity
    Eigen::MatrixXd I;

    /// Estimated states
    Eigen::VectorXd x_hat, x_hat_new;
};
#endif
