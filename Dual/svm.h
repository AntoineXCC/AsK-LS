#ifndef SVM_H
#define SVM_H

#if defined __GNUC__ || defined __APPLE__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

const double LAMBDA = 1;
const double RCOND_LIM = 1e-8;

/*
Solves linear system using either partial LU decomposition or Tikhonov
regularization depending of the condition number
Inputs are kernel matrix, label vector and flot gamma
Returns an Eigen Vector representing the support: 
[b_s, b_t, weight_t, weight_s]
*/
Eigen::VectorXf computeSupport(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, const float gamma);

#endif