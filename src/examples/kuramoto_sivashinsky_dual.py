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

    squareArray: np.array = None

    def __init__(self, nModes: float, epsilon1: float, epsilon2: float, epsilon3: float):
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        super().__init__(nModes)


    @override
    def initialize(self, uInit: Callable[[float, float], float]) -> None:
        squareArray = np.array([[
            n*n + m*m
        for n in range(self.nModes + 1)] for m in range(self.nModes + 1)])
        self.squareArray = squareArray

        vInit = lambda x, y: 0 # see if this works?

        return super().initialize([uInit, vInit])

    @override
    def generateLambdas(self) -> np.array:

        lambdas = -self.epsilon1 * self.squareArray

        return np.concat([lambdas, np.zeros((self.nModes + 1, self.nModes + 1))])
    
    @override
    def computeReaction(self, t: float, currentCoefficients: np.array) -> np.array:
        N: int = self.nModes
        alphas = currentCoefficients[0:(N+1), :]
        betas = currentCoefficients[(N+1):currentCoefficients.shape[0], :]

        # we don't exactly compute the convolution. But something close
        skewedConvolution = np.zeros((N+1, N+1))
        for n in range(N + 1):
            for m in range(N + 1):
                for k in range(-N, N+1):
                    for l in range(-N, N+1):
                        if abs(n - k) <= N and abs(m - l) <= N:
                            skewedConvolution[n, m] += (k * (n - k) + l * (m - l)) * alphas[abs(k), abs(l)] * alphas[abs(n - k), abs(m - l)]

        uReaction = self.epsilon2 * self.squareArray * betas + self.epsilon3 * skewedConvolution
        vReaction = - self.squareArray * alphas
        return np.concat([uReaction, vReaction])
        
if __name__ == "__main__":
    nModes: int = 20

    epsilon1: float = 1.0
    epsilon2: float = 1.0
    epsilon3: float = 1.0

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
    timesteps: int = 1000
    ksSimulator.runTimestepperAndCache(t0, h, timesteps)
    print("Ran Simulation. Proceeding to plot.")
    # print(f"Result from timestepper: {diffusion.getTimestepResult()}")

    ksSimulator.generatePlotForFunction(
        functionIndex = 0,
        nPartitionPerAxis = 100,
        title = "KS",
        saveAddr  = None,# f"movies/brusselator_u_modes_{nModes}_h_{h}_timesteps_{timesteps}_epsilon1_{epsilon1}_epsilon2_{epsilon2}_a_{a}_b_{b}",
        show = True,
        interval = 1,
        cmap = 'bwr',
        dynamicVs = True
    )