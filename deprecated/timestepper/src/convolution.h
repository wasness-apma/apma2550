# pragma once

#include <Eigen/Dense>
Eigen::MatrixXd cosineMatrix(int nrows, int ncolumns); 
Eigen::MatrixXd computeValuesOnCosineBasis(Eigen::MatrixXd coefficients, int basisSize);
Eigen::MatrixXd computeTripleConvolutionGivenTripleProducts(Eigen::MatrixXd tripleProduct, int originalBasisSize);