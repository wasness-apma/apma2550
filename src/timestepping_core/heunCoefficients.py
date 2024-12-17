import numpy as np
from typing import Callable, List, Tuple, Any, Optional
import copy
from src.core.require import require

# e^{-lambda h}hc(lambda h)
# limit is h as lambda \to 0
def expoEulerCoefficient(h: float, lambdas: np.array) -> float:
    return np.where(lambdas == 0, h, np.divide(1.0 - np.exp(-h*lambdas), lambdas, where=lambdas != 0))

# e^{-lambda h}hM(lambda h)
# limit is h/2 as lambda \to 0
def expoHeunSteppedCoefficient(h: float, lambdas: np.array) -> float:
    return np.where(lambdas == 0, h/2, np.divide(np.exp(-h*lambdas) + (lambdas * h - 1), h*lambdas*lambdas, where=lambdas != 0))

# e^{-lambda h}hL(lambda h)
# limit is h/2 as lambda \to 0
def expoHeunBaseCoefficient(h: float, lambdas: np.array) -> float:
    return np.where(lambdas == 0, h/2, np.divide(1 - (lambdas * h + 1)*np.exp(-h*lambdas), h*lambdas*lambdas, where=lambdas != 0))


if __name__ == "__main__": 
    for h in [1, 0.1, 0.01]:
        require(expoEulerCoefficient(h, 0) == h)
        require(expoHeunSteppedCoefficient(h, 0) == h/2)
        require(expoHeunBaseCoefficient(h, 0) == h/2)

        for l in [-2, -1, 1, 2]:
            require(expoEulerCoefficient(h, l) == (1.0 - np.exp(-h*l)) / l)
            require(expoHeunSteppedCoefficient(h, l) == ((np.exp(-h*l) + (l * h - 1)) / (h * l * l)))
            require(expoHeunBaseCoefficient(h, l) == ((1 - (l * h + 1) * np.exp(-h*l)) / (h * l * l)))

    print("Done!")
