import src.timestepping_core.heunCoefficients as heunCoefficients
from src.core.require import require
from typing import List, Callable
import numpy as np


#e^{-lambda h}(u_n + hc(lambda h)f(t_n, u_n))
def expoEulerStep(t_0: float, t_1: float, states: List[float], lambdas: np.array, reaction: Callable[[float, np.array], np.array]) -> List[float]:
    h = t_1 - t_0
    cs = heunCoefficients.expoEulerCoefficient(h, lambdas)

    # print(f"Reaction = {reaction(t_0, states)}")
    # print(f"Cs = {cs}")
    # print(f"Total Adjustment = {cs * reaction(t_0, states)}")
    # print(f"to Add to = np.exp(-lambdas * h) * states)")

    return (np.exp(-lambdas * h) * states) + cs * reaction(t_0, states)

# e^{-lambda h}(u_n + h(L(lambda h)f(t_n, u_n) + M(lambda h)f(t_n, u_n + Euler step)))
def expoHeunStep(t_0: float, t_1: float, states: List[float], lambdas: np.array, reaction: Callable[[float, np.array], np.array]) -> List[float]:
    h = t_1 - t_0
    # e^{-lambda h}(u_n + hc(lambda h)f(t_n, u_n))
    eulerSteppedState = expoEulerStep(t_0, t_1, states, lambdas, reaction)
    # e^{-lambda h}hL(lambda h)f(t_n, u_n)
    ls = heunCoefficients.expoHeunBaseCoefficient(h, lambdas)
    expoHeunBaseReaction = ls * reaction(t_0, states)
    # e^{-lambda h}hM(lambda h)f(t_n, e^{-lambda h}(u_n + hc(lambda h)f(t_n, u_n)))
    ms = heunCoefficients.expoHeunSteppedCoefficient(h, lambdas)
    expoHeunSteppedReaction = ms * reaction(t_1, eulerSteppedState)

    # print(f"Current State: {states}")
    # print(f"eulerSteppedState: {eulerSteppedState}")
    # print(f"expoHeunBaseReaction: {expoHeunBaseReaction}")
    # print(f"expoHeunSteppedReaction: {expoHeunSteppedReaction}")
    # toReturn = (np.exp(-lambdas * h) * states) + expoHeunBaseReaction + expoHeunSteppedReaction
    # print(f"toReturn: {toReturn}")
    
    return (np.exp(-lambdas * h) * states) + expoHeunBaseReaction + expoHeunSteppedReaction


if __name__ == "__main__":
    h = 0.5
    f = lambda t, coeffs: ((t+1) * coeffs + 1)
    lambdas = np.array([-2, 2, 4])
    states = np.array([1, 2, 3])

    expoEulerCalc = expoEulerStep(0, h, states, lambdas, f)
    expoEulerExpected = np.exp(-h * lambdas) * (states + h * ((np.exp(lambdas * h) - 1)/(lambdas * h)) * f(0, states))
    eulerDiff = expoEulerCalc - expoEulerExpected
    require(np.max(np.abs(eulerDiff)) < 0.000001 )

    expoHeunCalc = expoHeunStep(0, h, states, lambdas, f)
    expoHeunExpected = np.exp(-lambdas * h) * (states + h * 
    (
      ((np.exp(h*lambdas) - (lambdas * h + 1)) / (h*h*lambdas*lambdas)) * f(0, states)
        +
      ((1 + (lambdas * h - 1) * np.exp(h*lambdas)) / (h*h*lambdas*lambdas)) * f(h, expoEulerCalc)
    ))    
    heunDiff = expoHeunCalc - expoHeunExpected
    require(np.max(np.abs(heunDiff)) < 0.000001 )

    print("Done.")