import numpy as np
from typing import Callable, List, Tuple, Any
import copy

# utilize the fact that we get full weight of interior points, and half weight on endpoints to decrease queries
def trapezoidalQuadrature(f: Callable[[float], float], a: float, b: float, nPartitions: int):
    if nPartitions <= 0:
        raise "Need a positive number of partitions!"
    partitioned = np.linspace(a, b, nPartitions + 1)
    functionValues = np.array(list(map(f, partitioned)))
    h = (partitioned[1] - partitioned[0])
    return np.sum(functionValues[1:-1]*h) + 0.5 * (functionValues[0] + functionValues[-1])*h

def twoDimensionalTrapzoidalQuadrature(f: Callable[[float, float], float], xmin: float, xmax: float, ymin: float, ymax: float, nPartitions: int) -> float:

    xs = np.linspace(xmin, xmax, nPartitions + 1)
    ys = np.linspace(ymin, ymax, nPartitions + 1)

    h1 = xs[1] - xs[0]
    h2 = ys[1] - ys[0]

    tSum = 0
    for i in range(nPartitions + 1):
        iMult = 0.5 if i == 0 or i == nPartitions else 1
        for j in range(nPartitions + 1):
            jMult = 0.5 if j == 0 or j == nPartitions else 1
            tSum += h1 * h2 * iMult * jMult * f(xs[i], ys[j])


    return tSum

if __name__ == "__main__":

    for k in range(1, 10):
        n = int(np.power(2, k))
        trap = trapezoidalQuadrature(lambda x: x, 0, 1, n)
        if abs(trap - 0.5) > 0.00000001:
            raise f"Unexpected identity quadrature of {trap}"

    for k in range(1, 10):
        n = int(np.power(2, k))
        trap = trapezoidalQuadrature(lambda x: x*x, 0, 1, n)
        expected = (0.5 / n) + (n - 1) * (2 * n - 1) / (6 * n * n)
        if abs(trap - expected) > 0.00000001:
            raise f"Unexpected identity quadrature of {trap}"
        
    for k in range(1, 10):
        n = int(np.power(2, k))
        trap = twoDimensionalTrapzoidalQuadrature(lambda x, y: x*y, 0, 1, 0, 1, n)
        if abs(trap - 0.25) > 0.00000001:
            raise f"Unexpected identity quadrature of {trap}"
        
    for k in range(1, 10):
        n = int(np.power(2, k))
        trap = twoDimensionalTrapzoidalQuadrature(lambda x, y: x*x*y*y, 0, 1, 0, 1, n)
        print(trap)
        # if abs(trap - 0.25) > 0.00000001:
        #     raise f"Unexpected identity quadrature of {trap}"

    print("Success!")