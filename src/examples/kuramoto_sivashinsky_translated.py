from src.timestepping_core.genericExponentialTimestepper import GenericExpoHeunTimestepper
import src.fourier.convolution as convolution
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Tuple, override
import copy

class KuramotoSivashinsky(GenericExpoHeunTimestepper):
    epsilon1: float = None
    epsilon2: float = None
    epsilon3: float = None

    def __init__(self, nModes: float, epsilon1: float, epsilon2: float, epsilon3: float):
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        super().__init__(nModes)

    @override 
    def translatorFunction(self, n: int, m: int) -> float:
        if n == 0 and m == 0:
            return 1
        else:
            return 1.0 / (n**2 + m**2)

    @override
    def initialize(self, uInit: Callable[[float, float], float]) -> None:
        return super().initialize([uInit])

    @override
    def generateLambdas(self) -> np.array:
        lambdaArray = np.array([[
            -self.epsilon1 + self.epsilon2 * (n**2 + m**2) if n != 0 or m != 0 else 0
        for m in range(self.nModes + 1)] for n in range(self.nModes + 1)])

        return - self.epsilon1 * squareArray + self.epsilon2 * quarticArray
    
    @override
    def computeReaction(self, t: float, currentCoefficients: np.array) -> np.array:
        # we don't exactly compute the convolution. But something close
        N = self.nModes
        skewedConvolution = np.zeros((N+1, N+1))
        for n in range(N + 1):
            for m in range(N + 1):
                for k in range(-N, N+1):
                    for l in range(-N, N+1):
                        if abs(n - k) <= N and abs(m - l) <= N:
                            skewedConvolution[n, m] += (k * (n - k) + l * (m - l)) * currentCoefficients[abs(k), abs(l)] * currentCoefficients[abs(n - k), abs(m - l)]

        # print(f"Coeffs in Reaction: {currentCoefficients}")
        # print(f"Reaction Convolution: {skewedConvolution}")
        return self.epsilon3 * skewedConvolution
        
if __name__ == "__main__":
    nModes: int = 7

    epsilon1: float = 1.0
    epsilon2: float = 1.0
    epsilon3: float = 50.0

    ksSimulator = KuramotoSivashinsky(nModes, epsilon1 = epsilon1, epsilon2 = epsilon2, epsilon3 = epsilon3)
    
    # initialU = lambda x, y: 0 if ((x - np.pi/2)**2 + (y - np.pi/2)**2 ) < (np.pi/4)**2 else -45
    # initialU = lambda x, y: 1 if x**2 + y**2 < np.pi**2 / 4 else 0
    initialU = lambda x, y : 10 * (-0.5 + np.random.random())


    ksSimulator.initialize(initialU)
    # print(f"Lambdas: {diffusion.getLambdas()}")
    # print(f"Initial Coefficients: {diffusion.getInitialCoefficients()}")
    print("Initialized.")

    t0: float = 0
    h: float = 0.0001
    timesteps: int = 50
    ksSimulator.runTimestepperAndCache(t0, h, timesteps)
    print("Ran Simulation. Proceeding to plot.")
    # print(f"Result from timestepper: {diffusion.getTimestepResult()}")

    ksSimulator.generatePlotForFunction(
        functionIndex = 0,
        nPartitionPerAxis = 150,
        title = "KS",
        saveAddr  = None,# f"movies/brusselator_u_modes_{nModes}_h_{h}_timesteps_{timesteps}_epsilon1_{epsilon1}_epsilon2_{epsilon2}_a_{a}_b_{b}",
        show = True,
        interval = 0.1,
        cmap = 'winter',
        dynamicVs = True
    )