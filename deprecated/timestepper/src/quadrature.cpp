

double trapezoidalQuadrature(double (*f)(double x), double a, double b, int nPartitions) {
    if (nPartitions <= 0) {
        throw("Need a positive number of partitions");
    }
    if (a >= b) {
        throw ("Need a < b");
    }
    double h = (b - a) / (1.0 * nPartitions);
    double totalSum = 0;
    totalSum += 0.5 * h * f(a) + 0.5 * h * f(b);

    for (int i = 1; i <= nPartitions - 1; i++) {
        totalSum += h * f(a + h * i);
    };

    return totalSum;    
}