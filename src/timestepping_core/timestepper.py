import src.timestepping_core.heunSingleStep as heunSingleStep
from src.core.require import require
from typing import List, Callable, Optional, Tuple
import numpy as np


def computeAllPoints(
    t0: float, 
    h: float, 
    timesteps: int, 
    initialStates: np.array, 
    lambdas: np.array, 
    reaction: Callable[[float, np.array], np.array]
) -> List[Tuple[float, np.array]]:
    time = t0
    tracker = []
    currentResult = initialStates
    while len(tracker) < timesteps:
        tracker.append((
            time, 
            currentResult
        ))

        newResult = heunSingleStep.expoHeunStep(time, time + h, currentResult, lambdas, reaction)
        currentResult = newResult
        
        time = time + h
    return tracker

if __name__ == "__main__":
    # we test the callable u' = -lambda u + 1
    # so, f == 1.
    # we expect the formula u_n = e^{-nh lambda} u_0 + (1 - e^{-n h lambda})/lambda

    t0 = 0
    h = .01
    timesteps = 20
    initialStates = np.array([[-10, -5, 0], [0, 5, 10], [-20, 0, 20]])
    lambdas = np.array([[-10, 0, 10], [0, 10, 20], [-20, -5, 3]])
    reaction = lambda t, arr: (0 * arr + 1)

    allPoints = computeAllPoints(t0, h, timesteps, initialStates, lambdas, reaction)
    for n in range(len(allPoints)):
        (t, res) = allPoints[n]
        expected = np.exp(-n * h * lambdas) * initialStates + np.where(lambdas == 0, h*n, np.divide(1 - np.exp(-n*h*lambdas), lambdas, where=lambdas != 0))

        print(res)
        print(expected)
        require(np.max(np.abs(res - expected)) < 0.00001, throwMessage = f"Found discrepancy between {res} and {expected}")

    print("Done")