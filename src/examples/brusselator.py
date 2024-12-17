from src.timestepping_core.genericExponentialTimestepper import GenericExpoHeunTimestepper
import src.fourier.convolution as convolution
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Tuple, override
import copy

class Brusselator(GenericExpoHeunTimestepper):
    epsilon1: float = None
    epsilon2: float = None
    a: float = None
    b: float = None

    def __init__(self, nModes: float, epsilon1: float, epsilon2: float, a: float, b: float):
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.a = a
        self.b = b
        super().__init__(nModes)


    @override
    def initialize(self, uInit: Callable[[float, float], float], vInit: Callable[[float, float], float]) -> None:
        return super().initialize([uInit, vInit])

    @override
    def generateLambdas(self) -> np.array:
        squareArray = np.array([[
            n*n + m*m
        for n in range(self.nModes + 1)] for m in range(self.nModes + 1)])
        
        uLambdas = self.epsilon1 * squareArray + (1 + self.b)
        vLambdas = self.epsilon2 * squareArray
        return np.concat([uLambdas, vLambdas])
    
    @override
    def computeReaction(self, t: float, currentCoefficients: np.array) -> np.array:
        N: int = self.nModes
        alphas = currentCoefficients[0:(N+1), :]
        betas = currentCoefficients[(N+1):currentCoefficients.shape[0], :]

        convolve3 = convolution.convolveThree(alphas, alphas, betas)

        aReaction = copy.deepcopy(convolve3)
        aReaction[0, 0] += self.a

        bReaction = -convolve3 + self.b*alphas

        reaction = np.concat([aReaction, bReaction])

        return reaction
        
if __name__ == "__main__":
    nModes: int = 40

    for level in [0.001]:
        epsilon1: float = level * 0.05
        epsilon2: float = level * 1.

        a: float = 2
        b: float = 4

        brusselator = Brusselator(nModes, epsilon1 = epsilon1, epsilon2 = epsilon2, a = a, b = b)
        
        initialU = lambda x, y: a + 0.01 * np.random.random()# + x * 0 + 
        initialV = lambda x, y: (b/a) #+ y * 0 


        brusselator.initialize(initialU, initialV)
        # print(f"Lambdas: {diffusion.getLambdas()}")
        # print(f"Initial Coefficients: {diffusion.getInitialCoefficients()}")
        print("Initialized.")

        t0: float = 0
        h: float = 0.1
        timesteps: int = 15000

        title = lambda varName: f"movies/brusselator_{varName}_modes_{nModes}_h_{h}_timesteps_{timesteps}_level_{level}_epsilon1_{epsilon1}_epsilon2_{epsilon2}_a_{a}_b_{b}"
        

        brusselator.runTimestepperAndCache(t0, h, timesteps)
        print("Ran Simulation. Proceeding to plot.")
        # print(f"Result from timestepper: {diffusion.getTimestepResult()}")

        brusselator.generatePlotForFunction(
            functionIndex = 0,
            nPartitionPerAxis = 500,
            title = "u",
            saveAddr  = title("u"),
            show = False,
            interval = 25,
            stepsize = 10,
            cmap = 'winter',
            dynamicVs = True
        )

        brusselator.generatePlotForFunction(
            functionIndex = 1,
            nPartitionPerAxis = 500,
            title = "v",
            saveAddr  = title("v"),
            show = False,
            interval = 25,
            stepsize = 10,
            cmap = 'winter',
            dynamicVs = True
        )