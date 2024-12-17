import numpy as np
from typing import Callable, List, Tuple, Any
import numpy as np
# from abc import abstractmethod





if __name__ == "__main__":
    from src.core.quadrature import trapezoidalQuadrature, twoDimensionalTrapzoidalQuadrature
    from src.fourier.convolution import convolveTwo

    def deltaWithRounding(x, y) -> float:
        if np.abs(x - y) < 0.00001:
            return 1.0
        return 0.0

    # N = 10
    # for n1 in range(-N, N + 1):
    #     for n2 in range(-N, N + 1):
    #         for n3 in range(-N, N + 1):
    #             func = lambda x:  (np.cos(n1*x)) * (np.cos(n2*x)) * (np.cos(n3*x))
    #             quadrature = trapezoidalQuadrature(func, 0, np.pi, 175)
    #             guesstimate = 0.25 * np.pi * (deltaWithRounding(n1 + n2, n3) + deltaWithRounding(-n1 - n2, n3) + deltaWithRounding(n1 - n2, n3) + deltaWithRounding(-n1 + n2, n3))
    #             if np.abs(quadrature - guesstimate) > 0.01:
    #                 raise Exception(f"Bad values for COS at {(n1, n2, n3)}. Quad = {quadrature}. Guess = {guesstimate}")
                
    #             func = lambda x:  (np.sin(n1*x)) * (np.sin(n2*x)) * (np.cos(n3*x))    
    #             quadrature = trapezoidalQuadrature(func, 0, np.pi, 175)
    #             guesstimate = 0.25 * np.pi * (-deltaWithRounding(n1 + n2, n3) - deltaWithRounding(-n1 - n2, n3) + deltaWithRounding(n1 - n2, n3) + deltaWithRounding(-n1 + n2, n3))
    #             if np.abs(quadrature - guesstimate) > 0.01:
    #                 raise Exception(f"Bad values for SIN at {(n1, n2, n3)}. Quad = {quadrature}. Guess = {guesstimate}")
    #     print(f"Done for n1 = {n1}")

    # raise "Done"


    N = 2 # number of modes
    # coefficients = np.random.random((N + 1, N + 1))
    coefficients = np.array(
        [[ 4.85834005e-02, 1.48580604e-02, -1.69438073e-03],
        [-2.60269301e-02, -1.89023098e-05, -6.59959862e-03],
        [-2.84274198e-02,  2.70548058e-02, -4.98352108e-03]]
    )

    twoIfNot0 = lambda n: 1 if n == 0 else 2

    
    for n3 in range(N + 1):
        for m3 in range(N + 1):
            def functionWithApplication(x, y) -> float:
                s = 0
                for n1 in range(N + 1):
                    for n2 in range(N + 1):
                        for m1 in range(N + 1):
                            for m2 in range(N + 1):
                                s += coefficients[n1, m1] * coefficients[n2, m2] * (
                                    n1 * n2 * np.sin(n1 * x) * np.sin(n2 * x) * np.cos(m1 * y) * np.cos(m2 * y) +
                                    m1 * m2 * np.cos(n1 * x) * np.cos(n2 * x) * np.sin(m1 * y) * np.sin(m2 * y)
                                ) * twoIfNot0(n1) * twoIfNot0(n2) * twoIfNot0(m1) * twoIfNot0(m2) 
            
                return s * np.cos(n3 * x) * np.cos(m3 * y)
            
            quad = twoDimensionalTrapzoidalQuadrature(functionWithApplication, 0, np.pi, 0, np.pi, 50)

            # raw
            # estimate = 0
            # for n1 in range(N + 1):
            #     for n2 in range(N + 1):
            #         for m1 in range(N + 1):
            #             for m2 in range(N + 1):
            #                 estimate += coefficients[n1, m1] * coefficients[n2, m2] * twoIfNot0(n1) * twoIfNot0(n2) * twoIfNot0(m1) * twoIfNot0(m2) * (
            #                     n1 * n2 * (-deltaWithRounding(n1 + n2, n3) - deltaWithRounding(-n1 - n2, n3) + deltaWithRounding(n1 - n2, n3) + deltaWithRounding(-n1 + n2, n3)) * (+deltaWithRounding(m1 + m2, m3) + deltaWithRounding(-m1 - m2, m3) + deltaWithRounding(m1 - m2, m3) + deltaWithRounding(-m1 + m2, m3)) +
            #                     m1 * m2 * (+deltaWithRounding(n1 + n2, n3) + deltaWithRounding(-n1 - n2, n3) + deltaWithRounding(n1 - n2, n3) + deltaWithRounding(-n1 + n2, n3)) * (-deltaWithRounding(m1 + m2, m3) - deltaWithRounding(-m1 - m2, m3) + deltaWithRounding(m1 - m2, m3) + deltaWithRounding(-m1 + m2, m3))
            #                 )
            # estimate = estimate * (np.pi**2 / 16)

            # adding negatives via flipping
            # estimate = 0
            # for n1 in range(-N, N + 1):
            #     for n2 in range(-N, N + 1):
            #         for m1 in range(0, N + 1):
            #             for m2 in range(0, N + 1):
            #                 estimate += coefficients[abs(n1), abs(m1)] * coefficients[abs(n2), abs(m2)] * twoIfNot0(m1) * twoIfNot0(m2) * (
            #                      n1 * n2 * deltaWithRounding(n1 + n2, n3) * (+deltaWithRounding(m1 + m2, m3) + deltaWithRounding(-m1 - m2, m3) + deltaWithRounding(m1 - m2, m3) + deltaWithRounding(-m1 + m2, m3))
            #                 )
            # for n1 in range(0, N + 1):
            #     for n2 in range(0, N + 1):
            #         for m1 in range(-N, N + 1):
            #             for m2 in range(-N, N + 1):
            #                 estimate += coefficients[abs(n1), abs(m1)] * coefficients[abs(n2), abs(m2)] * twoIfNot0(n1) * twoIfNot0(n2) * (
            #                     m1 * m2 * (+deltaWithRounding(n1 + n2, n3) + deltaWithRounding(-n1 - n2, n3) + deltaWithRounding(n1 - n2, n3) + deltaWithRounding(-n1 + n2, n3)) * deltaWithRounding(m1 + m2, m3)
            #                 )
            # estimate = -(np.pi**2 / 4) *  estimate
            
            # simplifying m terms by symmetry
            # estimate = 0
            # for n1 in range(-N, N + 1):
            #     for n2 in range(-N, N + 1):
            #         for m1 in range(-N, N + 1):
            #             for m2 in range(-N, N + 1):
            #                 estimate += coefficients[abs(n1), abs(m1)] * coefficients[abs(n2), abs(m2)] * (
            #                     n1 * n2 * deltaWithRounding(n1 + n2, n3) * deltaWithRounding(m1 + m2, m3) +
            #                     m1 * m2 * deltaWithRounding(n1 + n2, n3) * deltaWithRounding(m1 + m2, m3)
            #                 )
            # estimate = -(np.pi**2) * estimate
    
            # the smartest estimate we have: pseudo-convolution.
            # coeffRows = np.array([[
            #     n * coefficients[n, m]
            # for m in range(N+1)] for n in range(N + 1)])
            # coeffCols = np.array([[
            #     m * coefficients[n, m]
            # for m in range(N+1)] for n in range(N + 1)])
            convolved = 0
            for k in range(-coefficients.shape[0] + 1, coefficients.shape[0]):
                for l in range(-coefficients.shape[1] + 1, coefficients.shape[1]):
                    if abs(n3 - k) < coefficients.shape[0] and abs(m3 - l) < coefficients.shape[1]:
                        convolved += (k * (n3 - k) + l * (m3 - l)) * coefficients[abs(k), abs(l)] * coefficients[abs(n3 - k), abs(m3 - l)]
            print(f"{n3, m3}: {convolved}")
            estimate = convolved * -(np.pi**2)

            # estimate = (- np.pi**2 / 16) * (convolveTwo(coeffRows, coeffRows)  + convolveTwo(coeffCols, coeffCols))[n3, m3

            if np.abs(quad - estimate) > 0.0001:
                print(f"Failure at {n3}, {m3}")
                print(f"Quad: {quad}")
                print(f"Estimate: {estimate}")
                print(coefficients)
                raise "Failure."
            # else:
                # print(f"Success at {n3}, {m3}")
                # print(f"Quad: {quad}")
                # print(f"Estimate: {estimate}")
    print("Success")




    # print("Next thing")

    # coefficientsA = np.random.random((N + 1, N + 1))
    # coefficientsB = np.random.random((N  + 1, N + 1))

    # def funcA(x: float, y: float):
    #     s = 0
    #     for n in range(N + 1):
    #         for m in range(N + 1):
    #             s += coefficientsA[n, m] * np.cos(n * x) * np.cos(m * y) * twoIfNot0(n) * twoIfNot0(m)
    #     return s

    # def funcB(x: float, y: float):
    #     s = 0
    #     for n in range(N + 1):
    #         for m in range(N + 1):
    #             s += coefficientsB[n, m] * np.cos(n * x) * np.cos(m * y) * twoIfNot0(n) * twoIfNot0(m)
    #     return s
    

    # n1 = 2
    # n2 = 1
    # quad = twoDimensionalTrapzoidalQuadrature(lambda x, y: funcA(x, y) * funcB(x, y) * np.cos(n1 * x) * np.cos(n2 * y), 0, np.pi, 0, np.pi, 50)

    # conv = 0
    # for k in range(-N, N + 1):
    #     for l in range(-N, N + 1):
    #         if abs(n1 - k) <= N and abs(n2 - l) <= N:
    #             conv += coefficientsA[abs(k), abs(l)] * coefficientsB[abs(n1 - k), abs(n2 - l)]
    # conv = conv * (np.pi * np.pi)

    # print(quad)
    # print(conv)


