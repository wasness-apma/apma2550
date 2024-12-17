#include <Eigen/Dense>
#include <math.h>
#include <cmath>

// #include <iostream>

// generate an (N+1) \times M matrix A, such that the indices go from
// (0, 1), ... (N+1, M), where
// we have A[k, n] = cos(k * (2 * n - 1) * np.pi / (2 * ncolumns))
Eigen::MatrixXd cosineMatrix(int nrows, int ncolumns) {
    Eigen::MatrixXd cosMat(nrows + 1, ncolumns);
    for (int row = 0; row <= nrows; row++) {
        for (int col = 1; col <= ncolumns; col ++) {
            cosMat(row, col-1) = cos((1.0 * row) * (2.0 * col - 1) * M_PI / (2.0 * ncolumns));
        }
    }
    return cosMat;
}

/*
Given a matrix of coefficients alpha_{ij} for the decomposition

u(x, y) = sum'_{i, j = 0}^N alpha_{ij} cos(x)cos(y),

find the matrix of u(s_n, s_m) where s_m = pi (2m - 1) / 2M for m = 1, ..., M
*/
Eigen::MatrixXd computeValuesOnCosineBasis(Eigen::MatrixXd coefficients, int basisSize) {
    if (coefficients.rows() != coefficients.cols()) {
        throw ("Bad coefficient size");
    }
    int N = coefficients.rows() - 1;

    Eigen::MatrixXd halvingMatrix = Eigen::MatrixXd::Identity(N + 1, N + 1);
    halvingMatrix(0, 0) = 0.5;
    // std::cout << "Halving Matrix:\n" << halvingMatrix << std::endl;

    Eigen::MatrixXd cosineMatrixCoeffBasisWithHalves = halvingMatrix * cosineMatrix(N, basisSize);
    // std::cout << "Cosine Matrix On Coeff Basis:\n" << cosineMatrixCoeffBasisWithHalves << std::endl;

    return 4 * cosineMatrixCoeffBasisWithHalves.transpose() * coefficients * cosineMatrixCoeffBasisWithHalves;
}

Eigen::MatrixXd computeTripleConvolutionGivenTripleProducts(Eigen::MatrixXd tripleProduct, int originalBasisSize) {
    if (tripleProduct.rows() != tripleProduct.cols()) {
        throw ("Bad coefficient size");
    }
    if (tripleProduct.rows() < 2 * originalBasisSize - 1) {
        throw ("Bad original basis size");
    }
    int M = tripleProduct.rows();

    Eigen::MatrixXd cosineMatrixForConvolutionOverM = cosineMatrix(originalBasisSize, M) / (1.0 * M);
    return cosineMatrixForConvolutionOverM * tripleProduct * cosineMatrixForConvolutionOverM.transpose();
}