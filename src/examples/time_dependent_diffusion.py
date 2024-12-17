from src.timestepping_core.genericExponentialTimestepper import GenericExpoHeunTimestepper
import src.fourier.convolution as convolution
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Tuple, override
import copy

class Diffusion(GenericExpoHeunTimestepper):
    epsilon: float = None

    def __init__(self, nModes: float, epsilon: float,):
        self.epsilon = epsilon
        super().__init__(nModes)


    @override
    def initialize(self, uInit: Callable[[float, float], float]) -> None:
        return super().initialize([uInit])

    @override
    def generateLambdas(self) -> np.array:
        squareArray = np.array([[
            n*n + m*m
        for n in range(self.nModes + 1)] for m in range(self.nModes + 1)])
        return self.epsilon * squareArray
    
    @override
    def computeReaction(self, t: float, currentCoefficients: np.array) -> np.array:
        return (10/(t+1)) + 0 * currentCoefficients
        
if __name__ == "__main__":
    nModes: int = 20

    epsilon: float = 1.0/100

    diffusion = Diffusion(nModes, epsilon = epsilon)
    
    initialU = lambda x, y: 10 if np.abs(x - np.pi / 2) < np.pi / 4 and np.abs(y - np.pi / 2) < np.pi / 4 else 0
    # initialU = lambda x, y: 10 if np.abs(x) < np.pi / 2 and np.abs(y) < np.pi / 2 else 0


    diffusion.initialize(initialU)
    print("Initialized.")

    t0: float = 0
    h: float = 0.1
    timesteps: int = 1000
    diffusion.runTimestepperAndCache(t0, h, timesteps)
    print("Ran Simulation. Proceeding to plot.")
    # print(f"Result from timestepper: {diffusion.getTimestepResult()}")

    diffusion.generatePlotForFunction(
        functionIndex = 0,
        nPartitionPerAxis = 100,
        title = "u",
        saveAddr  = None,
        show = True,
        interval = 100,
        cmap = 'bwr',#'winter',
        dynamicVs = True
    )