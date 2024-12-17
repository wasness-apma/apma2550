from src.timestepping_core.genericExponentialTimestepper import GenericExpoHeunTimestepper
import src.fourier.convolution as convolution
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Tuple, override
import copy

class FizhughNagumo(GenericExpoHeunTimestepper):
    epsilon_u: float = None
    epsilon_v: float = None
    D: float = None
    a_v: float = None
    a_z: float = None

    def __init__(self, nModes: float, epsilon_s: float, epsilon_v: float, D: float, a_v: float, a_z: float):
        self.epsilon_s = epsilon_s
        self.epsilon_v = epsilon_v
        self.D = D
        self.a_v = a_v
        self.a_z = a_z
        super().__init__(nModes)


    @override
    def initialize(self, uInit: Callable[[float, float], float], vInit: Callable[[float, float], float]) -> None:
        return super().initialize([uInit, vInit])

    @override
    def generateLambdas(self) -> np.array:
        squareArray = np.array([[
            n*n + m*m 
        for n in range(self.nModes + 1)] for m in range(self.nModes + 1)])
        
        uLambdas = self.epsilon_s * squareArray - 1
        vLambdas = self.epsilon_s * self.D * squareArray + self.epsilon_v * self.a_v
        return np.concat([uLambdas, vLambdas])
    
    @override
    def computeReaction(self, t: float, currentCoefficients: np.array) -> np.array:
        N: int = self.nModes
        alphas = currentCoefficients[0:(N+1), :]
        betas = currentCoefficients[(N+1):currentCoefficients.shape[0], :]

        convolve3u = convolution.convolveThree(alphas, alphas, alphas)

        aReaction = - copy.deepcopy(convolve3u) - betas

        bReaction = self.epsilon_v * copy.deepcopy(alphas)
        bReaction[0, 0] -= self.epsilon_v * self.a_z

        reaction = np.concat([aReaction, bReaction])

        return reaction
        
if __name__ == "__main__":
    nModes: int = 20

    epsilon_s = 0.00001
    # D = 26. 
    D = 20.

    epsilon_v = 0.2
    # epsilon_v = 0.5

    a_v = 0.01
    # a_v = 1.

    a_z = -0.1

    # for m in [3, 4, 6]:
    fitzhugh_nagumo = FizhughNagumo(nModes, epsilon_s = epsilon_s,  epsilon_v = epsilon_v, D = D, a_v = a_v, a_z = a_z)
        
    # initialU = lambda x, y: np.cos(m * x) * np.cos(m * y) 
    # initialU = lambda x, y: -0.46415901110464597 + 0.1 * np.random.random()
    # initialU = lambda x, y: -0.10099969706149653 + 0.01 * np.random.random()
    initialU = lambda x, y: 1 if (x - np.pi/2)**2 + (y - np.pi/2)**2 < (np.pi / 16)**2 else 0.01 * np.random.random()
    initialV = lambda x, y: 0
    # initialV = lambda x, y: -0.364159011104646 + 0.1 * np.random.random()
    # initialV = lambda x, y: -0.0999697061496524 + 0.01 * np.random.random()

    fitzhugh_nagumo.initialize(initialU, initialV)
    # print(f"Lambdas: {diffusion.getLambdas()}")
    # print(f"Initial Coefficients: {diffusion.getInitialCoefficients()}")
    print("Initialized.")

    t0: float = 0
    h: float = 0.1
    timesteps: int = 1000

    title = lambda myVar: f"movies/fitzhugh_nagumo_circle_{myVar}_{nModes}_h_{h}_timesteps_{timesteps}_level_{level}_epsilon_u_{epsilon_u}_D_{D}_epsilon_v_{epsilon_v}_a_v_{a_v}_a_z_{a_z}"

    fitzhugh_nagumo.runTimestepperAndCache(t0, h, timesteps)
    print("Ran Simulation. Proceeding to plot.")
    # print(f"Result from timestepper: {fitzhugh_nagumo.getTimestepResult()}")

    fitzhugh_nagumo.generatePlotForFunction(
        functionIndex = 0,
        nPartitionPerAxis = 250,
        title = "u",
        saveAddr  = title("u"),
        show = False,
        stepsize = 5,
        interval = 75,
        cmap = 'winter',#'bwr'
        vmin = -1,
        vmax = 1.
    )

    fitzhugh_nagumo.generatePlotForFunction(
        functionIndex = 0,
        nPartitionPerAxis = 250,
        title = "v",
        saveAddr  = title("v"),
        show = False,
        stepsize = 5,
        interval = 75,
        cmap = 'winter',#'bwr'
        vmin = -1,
        vmax = 1.
    )