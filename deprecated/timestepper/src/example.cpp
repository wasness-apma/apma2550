#include "convolution.h"
#include <Eigen/Dense>
#include <iostream>
#include "quadrature.h"
#include <math.h>

double identity(double x) {
    return x;
}

double square(double x) {
    return x*x;
}

int main() 
{
    
    //Code to test convolution

    Eigen::MatrixXd m = cosineMatrix(5, 5);
    std::cout << "Cosine Matrix(4, 4):\n" << m << std::endl;

    Eigen::MatrixXd coeffMatrix(3, 3);
    coeffMatrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    std::cout << "Coeff Matrix:\n" << coeffMatrix << std::endl;

    Eigen::MatrixXd coeffsOnBasis = computeValuesOnCosineBasis(coeffMatrix, 5);
    std::cout << "Coeffs on Basis of M = 4:\n" << coeffsOnBasis << std::endl;

    Eigen::MatrixXd tripleProduct = coeffsOnBasis.cwiseProduct(coeffsOnBasis).cwiseProduct(coeffsOnBasis);
    std::cout << "Component Wise Triple Products:\n" << tripleProduct << std::endl;

    Eigen::MatrixXd tripleConvolution = computeTripleConvolutionGivenTripleProducts(tripleProduct, 3);
    std::cout << "Component Wise Triple Convolution:\n" << tripleConvolution << std::endl;
    

   /*
   quadrature test 

    double quadratureIdentity = trapezoidalQuadrature(identity, 0, 1, 10);
    std::cout << "Trapezoidal Quadrature for Identity:\n" << quadratureIdentity << std::endl;

    double quadratureSquare = trapezoidalQuadrature(square, 0, 1, 10);
    std::cout << "Trapezoidal Quadrature for Square:\n" << quadratureSquare << std::endl;
    */

   /*
   Heun coefficient test
   */

   

    return 0;
}