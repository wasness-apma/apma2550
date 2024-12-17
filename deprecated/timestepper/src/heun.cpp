#include <math.h>
#include <functional>
#include <Eigen/Dense>

// e^(-h lambda) * h * c(h lambda)
double calcCMaybe(double h, double lambda) {
    if (lambda == 0) {
        return h;
    } else {
        return (1 - std::exp(lambda * h)) / lambda;
    }
}
// Eigen::MatrixXd expoEulerCoefficient(double h, Eigen::MatrixXd lambdas) {
//     auto calcCMaybeWithH = std::bind(calcCMaybe, h, std::placeholders::_1);
//     return lambdas.unaryExpr(calcCMaybeWithH);
// }