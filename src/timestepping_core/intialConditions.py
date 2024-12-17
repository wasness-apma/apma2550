from src.core.require import require
from typing import List, Callable, Optional, Tuple
import numpy as np
import src.core.quadrature as quadrature

def computeUinitOn2dGrid(nModes: int, gridFunction: Callable[[float, float], float], constantTranslator: Callable[[int, int], float] = lambda x: x) -> np.array:
    initialStates = [[
        quadrature.twoDimensionalTrapzoidalQuadrature(
            f = 
            (lambda _xmode, _ymode: 
             lambda x, y: gridFunction(x, y) * np.cos(_xmode * x) * np.cos(_ymode * y) / constantTranslator(_xmode, _ymode)
             )(xmode, ymode),
            xmin = 0,
            xmax = np.pi,
            ymin = 0,
            ymax = np.pi,
            nPartitions = 4 * nModes + 1
        ) / (np.pi ** 2)
    for ymode in range(nModes + 1)] for xmode in range(nModes + 1)]
    return(np.array(initialStates))

if __name__ == "__main__":
    # test constant function
    constantOutput = computeUinitOn2dGrid(10, lambda x, y: 1)
    constantExpectedOutput = np.zeros((11, 11))
    constantExpectedOutput[0, 0] = 1
    require(np.max(np.abs(constantOutput - constantExpectedOutput)) < 0.00001)

    testN = 3
    testCoefficients = np.array([
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    testFunction = lambda x, y: np.sum([
        4 * (0.5 if xmode == 0 else 1) * (0.5 if ymode == 0 else 1) * testCoefficients[xmode, ymode] * np.cos(xmode * x) * np.cos(ymode * y) 
        for xmode in range(testN + 1) for ymode in range(testN + 1)
    ])
    genericOutput = computeUinitOn2dGrid (testN, testFunction)
    require(np.max(np.abs(genericOutput - testCoefficients)) < 0.00001)

    print("Success")