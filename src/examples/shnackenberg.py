from src.timestepping_core.genericExponentialTimestepper import GenericExpoHeunTimestepper
import src.fourier.convolution as convolution
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Tuple, override
import copy

class Shnackenberg(GenericExpoHeunTimestepper):
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
        
        uLambdas = self.epsilon1 * squareArray + 1
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

        bReaction = -copy.deepcopy(convolve3)
        bReaction[0, 0] += self.b

        reaction = np.concat([aReaction, bReaction])

        return reaction
        
if __name__ == "__main__":
    nModes: int = 30

    level = 0.001
    epsilon1: float = 1.0 * level
    epsilon2: float = 30.0 * level # 100.0

    a: float = 0.01 # 0.05
    b: float = 2.0 # 0.5

    shnackenberg = Shnackenberg(nModes, epsilon1 = epsilon1, epsilon2 = epsilon2, a = a, b = b)
    
    # initialU = lambda x, y: 1 if (x - np.pi/2)**2 + (y - np.pi/2)**2 < (np.pi / 8)**2 else a + b
    initialU = lambda x, y: a + b + 0.01 * np.random.random() # + 0.3 * (a + b) * np.cos(3 * x) * np.cos(4 * y)
    initialV = lambda x, y: b / ((a + b)**2)


    shnackenberg.initialize(initialU, initialV)
    # print(f"Lambdas: {diffusion.getLambdas()}")
    # print(f"Initial Coefficients: {diffusion.getInitialCoefficients()}")
    print("Initialized.")

    t0: float = 0
    h: float = 0.1
    timesteps: int = 15000

    title = f"movies/shnackenberg_u_modes_{nModes}_h_{h}_timesteps_{timesteps}_level_{level}_epsilon1_{epsilon1}_epsilon2_{epsilon2}_a_{a}_b_{b}"

    shnackenberg.runTimestepperAndCache(t0, h, timesteps)
    print("Ran Simulation. Proceeding to plot.")
    # print(f"Result from timestepper: {diffusion.getTimestepResult()}")

    shnackenberg.generatePlotForFunction(
        functionIndex = 0,
        nPartitionPerAxis = 250,
        title = "u",
        saveAddr  = title,
        show = True,
        interval = 25,
        cmap = 'winter',
        stepsize = 10,
        # dynamicVs = True
        vmin = 0.0,
        vmax = 3.5,
    )

    # shnackenberg.generatePlotForFunction(
    #     functionIndex = 1,
    #     nPartitionPerAxis = 200,
    #     title = "v",
    #     saveAddr  = None,#"movies/brusselator_v_modes_{nModes}_h_{h}_timesteps_{timesteps}_epsilon1_{epsilon1}_epsilon2_{epsilon2}_a_{a}_b_{b}",
    #     show = True,
    #     interval = 5,
    #     cmap = 'winter',
    #     dynamicVs = True
    # )