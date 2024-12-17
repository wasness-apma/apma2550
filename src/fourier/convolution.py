import numpy as np
from typing import Callable, List, Tuple, Any
import copy
import scipy

cosineMatrixCache = {}
def getCosineMatrix(N, M):
    if (N, M) in cosineMatrixCache:
        return cosineMatrixCache[(N, M)]
    cosineMatrix = np.zeros((N + 1, M))
    for k in range(N + 1):
        for n in range(M):
            cosineMatrix[k, n] = np.cos(k * (2 * (n + 1) - 1) * np.pi / (2 * M))
    cosineMatrixCache[(N, M)] = cosineMatrix
    return cosineMatrix

def computeSnFunctionOnCosineBasis(coefficients: np.array, M: int) -> np.array:
    if coefficients.shape[0] != coefficients.shape[1]:
        raise Exception(f"Improper shape! Cannot accept dimensions {coefficients.shape}.")
    N = coefficients.shape[0] - 1

    coefficientsWithHalves = 1.0 * copy.deepcopy(coefficients)
    coefficientsWithHalves[:, 0] = 0.5 * coefficientsWithHalves[:, 0]
    coefficientsWithHalves[0, :] = 0.5 * coefficientsWithHalves[0, :]

    cosineMatrix = getCosineMatrix(N, M)

    Omega = 2*np.matmul(coefficientsWithHalves, cosineMatrix)
    U = 2 * np.matmul(np.transpose(cosineMatrix), Omega)
    return U

def computeConvolutionFunctionOnCosineBasisGivenTripleProducts(products: np.array, N: int) -> np.array:
    if products.shape[0] != products.shape[1]:
        raise Exception(f"Improper shape! Cannot accept dimensions {products.shape}.")
    M = products.shape[0]

    cosineMatrix = getCosineMatrix(N, M)

    Sigma = np.matmul(cosineMatrix, np.transpose(products) / (1.0 * M))
    Conv = np.matmul(cosineMatrix, np.transpose(Sigma) / (1.0 * M))
    return Conv

def convolveThree(a: np.array, b: np.array, c: np.array) -> np.array:
    if a.shape != b.shape or a.shape != c.shape:
        raise f"Bad shapes! {a.shape}, {b.shape}, {c.shape}"
    if a.shape[0] != a.shape[1]:
        raise f"Not square! {a.shape}"

    N = a.shape[0] - 1
    M = 2 * N + 1
    aOnBasis = computeSnFunctionOnCosineBasis(a, M)
    bOnBasis = computeSnFunctionOnCosineBasis(b, M)
    cOnBasis = computeSnFunctionOnCosineBasis(c, M)
    tripleProduct = aOnBasis * bOnBasis * cOnBasis
    convolved = computeConvolutionFunctionOnCosineBasisGivenTripleProducts(tripleProduct, N)
    return convolved    

#convolve two matrices.
def convolveTwo(a: np.array, b: np.array) -> np.array:
    if a.shape != b.shape:
        raise f"Bad shapes! {a.shape}, {b.shape}"
    if a.shape[0] != a.shape[1]:
        raise f"Not square! {a.shape}"

    convolved = np.zeros(a.shape)
    for n in range(a.shape[0]):
        for m in range(a.shape[1]):
            for k in range(-a.shape[0] + 1, a.shape[0]):
                for l in range(-a.shape[1] + 1, a.shape[1]):
                    if abs(n - k) < a.shape[0] and abs(m - l) < a.shape[1]:
                        convolved[n, m] += a[abs(k), abs(l)] * b[abs(n - k), abs(m - l)]
    # N = a.shape[0] - 1
    # M = 2 * N + 1
    # aOnBasis = computeSnFunctionOnCosineBasis(a, M)
    # bOnBasis = computeSnFunctionOnCosineBasis(b, M)
    # tripleProduct = aOnBasis * bOnBasis
    # convolved = computeConvolutionFunctionOnCosineBasisGivenTripleProducts(tripleProduct, N)
    return convolved 


if __name__ == "__main__":
    # an example
    # alpha = np.array([
    #     [1.0074, -0.000428724, 0.000211051, -6.53045e-05],
    #     [0.000376411, -0.000401971, 0.000208362, 5.69628e-05],
    #     [-0.000314326, -1.09382e-05, -0.00021266, -0.000307548],
    #     [5.76987e-05, 0.00033859, 0.00018819, -6.93557e-05]
    # ])

    # beta = np.array([
    #     [3.00071, -2.54683e-05, -0.000151995, 8.02481e-06],
    #     [0.000390039, 0.000112105, -0.000236913, 0.000208895],
    #     [7.67416e-05, -0.000191853, 0.000203555, 0.000240355],
    #     [-5.31369e-06, -0.000105669, -5.95777e-05, 3.1128e-06]
    # ])

    # convolved = convolveThree(alpha, alpha, beta)
    # expected = np.array([
    #     [ 3.04529216e+00, -2.61940291e-03,  1.12473655e-03, -3.85460486e-04],
    #     [ 2.67196688e-03, -2.31904231e-03,  1.01965224e-03,  5.54450046e-04], 
    #     [-1.82152203e-03, -2.60428374e-04, -1.07770718e-03, -1.61530207e-03],
    #     [ 3.41099294e-04,  1.93976290e-03,  1.07654684e-03, -4.17111111e-04]
    # ])

    # if np.max(np.abs(convolved - expected)) > 0.0000001:
    #     raise "Unexpected Disagreement"
    # print("Got expected convolution value!")

    a = np.array([[1, 2], [3, 4]])
    print(convolveTwo(a, a))

    s = 0

    n3 = 1
    m3 = 1
    N = a.shape[0]
    for n1 in range(-N + 1, N):
        for n2 in range(-N + 1, N):
            for m1 in range(-N + 1, N):
                for m2 in range(-N + 1, N):
                    if n1 + n2 == n3 and m1 + m2 == m3:
                        s += a[abs(n1), abs(m1)] * a[abs(n2), abs(m2)]
    print(s)

    
