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
        return 0 * currentCoefficients
        
if __name__ == "__main__":
    nModes: int = 20

    for level in [1., 0.1, 0.001]:
        initialU_square = lambda x, y: 1 if np.abs(x - np.pi / 2) < np.pi / 4 and np.abs(y - np.pi / 2) < np.pi / 4 else 0
        initialU_periodic = lambda x, y: np.cos(3 * x) * np.cos(3 * y) 

        initialDict = {
            "square": initialU_square,
            "periodic":initialU_periodic
        }
        for name in initialDict:
            initialU = initialDict[name]

            epsilon: float = 1. * level

            diffusion = Diffusion(nModes, epsilon = epsilon)

            diffusion.initialize(initialU)
            print("Initialized.")

            t0: float = 0
            h: float = 0.1
            timesteps: int = 20 if level > 0.5 else (200 if level > 0.05 else 15000)

            title = f"movies/diffusion_{name}_modes_{nModes}_h_{h}_timesteps_{timesteps}_level_{level}_epsilon_{epsilon}"

            diffusion.runTimestepperAndCache(t0, h, timesteps)
            print("Ran Simulation. Proceeding to plot.")

            diffusion.generatePlotForFunction(
                functionIndex = 0,
                nPartitionPerAxis = 125,
                title = "u",
                saveAddr  = title,
                show = False,
                interval = 100 if level > 0.05 else 25,
                cmap = 'winter',
                stepsize = 1 if level > 0.005 else 10,
                dynamicVs = True
            )