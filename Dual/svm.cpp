#include "svm.h"
#include <iostream>

Eigen::VectorXf computeSupport(const Eigen::MatrixXf& X, const Eigen::VectorXf& Y, const float gamma) {

    // initializing structures
    int n = X.rows();
    Eigen::MatrixXf a = Eigen::MatrixXf::Zero(2 + 2 * n, 2 + 2 * n);
    Eigen::VectorXf b(2 * n + 2);
    Eigen::VectorXf labels = Y;
    Eigen::MatrixXf H = X.cwiseProduct(labels * labels.transpose());
    Eigen::VectorXf out;

    // construction of b vector
    b << Eigen::VectorXf::Zero(2), Eigen::VectorXf::Ones(n), Eigen::VectorXf::Ones(n);

    // defining matrix data points
    // A[0, 2:2+n] = A[1, 2+n:] = Y
    a.row(0).segment(2, n) = labels;
    a.row(1).segment(2 + n, n) = labels;

    // A[2:2+n, 0] = A[2+n:, 1] = Y
    a.block(2, 0, n, 1) = labels;
    a.block(2 + n, 1, n, 1) = labels;

    // A[2:2+n, 2:2+n] = A[2+n:,2+n:] = np.eye(n)/self.gam
    a.block(2, 2, n, n) = Eigen::MatrixXf::Identity(n, n) / gamma;
    a.block(2 + n, 2 + n, n, n) = Eigen::MatrixXf::Identity(n, n) / gamma;

    // A[2:2+n, 2+n:] = H
    a.block(2, 2 + n, n, H.cols()) = H;

    // A[2+n:, 2:2+n] = H.T
    a.block(2 + n, 2, H.cols(), n) = H.transpose();

    Eigen::PartialPivLU<Eigen::MatrixXf> LU_decomp(a);
    // Apply either LU decomposition or Tikhonov Regularization depending of the condition number

    if (LU_decomp.rcond()>RCOND_LIM) {
        out = LU_decomp.solve(b);
    } else {
        Eigen::MatrixXf a_transpose = a.transpose();
        Eigen::MatrixXf eye = LAMBDA * Eigen::MatrixXf::Identity(2*n + 2, 2*n + 2);
        out = (a_transpose * a + eye).lu().solve(a_transpose * b);
    }
    out.segment(2, n) = out.segment(2, n).array() * Y.array();
    out.segment(2+n, n) = out.segment(2+n, n).array() * Y.array();
    return out;
}
