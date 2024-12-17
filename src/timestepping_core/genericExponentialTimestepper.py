from numpy._core.multiarray import array as array
from src.core.require import require
from src.timestepping_core.intialConditions import computeUinitOn2dGrid
import src.timestepping_core.timestepper as timestepper
from typing import List, Callable, Optional, Tuple, override
import numpy as np
from abc import ABC, abstractmethod
from src.plotting.heatmapMovie import plotXyVectorizedHeatmapMovieOnGrid

# a generic method to compute a reaction-diffusion solution with Neumann boundary conditions on [0, pi] x [0, pi]
# allows for multiple functions to be used, but they need to combine into a simple ODE system of standard form
# then, we can "unpick" them,


class GenericExpoHeunTimestepper:

    @abstractmethod
    def generateLambdas(self) -> np.array:
        pass

    @abstractmethod
    def computeReaction(self, t: float, currentCoefficients: np.array) -> np.array:
        pass

    nModes: float = None
    nFunctions: int = None

    lambdas: np.array = None
    initialCoefficients: np.array = None

    timestepResult: List[Tuple[float, np.array]] = None

    translatorMatrix: np.array

    def __init__(self, nModes: float):
        self.nModes = nModes

    # idea is that our coefficients might be of the form c_{nm}alpha_{nm}cos(nx)cos(my)
    # to, potentially, account for numerical instability
    def translatorFunction(self, n: int, m: int) -> float:
        return 1.0


    def initialize(self, initialConditions: List[Callable[[float, float], float]]) -> None: 
        self.nFunctions = len(initialConditions)
        allInitialConditions = []
        for initialConditionFunction in initialConditions:
            initialCoefficentsForFunction = computeUinitOn2dGrid(self.nModes, initialConditionFunction, constantTranslator = self.translatorFunction)
            allInitialConditions.append(initialCoefficentsForFunction)
        self.initialCoefficients = np.concat(allInitialConditions)

        self.translatorMatrix = np.array([[
            self.translatorFunction(n, m)
        for m in range(self.nModes + 1)] for n in range(self.nModes + 1)])

        self.lambdas = self.generateLambdas()

    def getInitialCoefficients(self) -> np.array:
        if self.initialCoefficients is None:
            raise "No initial coefficients"
        return self.initialCoefficients
    
    def getLambdas(self) -> np.array:
        if self.lambdas is None:
            raise "No lambdas"
        return self.lambdas


    # t0: float, 
    # h: float, 
    # timesteps: int, 
    # initialStates: np.array, 
    # lambdas: np.array, 
    # reaction: Callable[[float, np.array], np.array]
    def runTimestepperAndCache(self, t0: float, h: float, timesteps: int) -> List[Tuple[float, np.array]]:
        self.timestepResult = timestepper.computeAllPoints(
            t0 = t0,
            h = h,
            timesteps = timesteps,
            initialStates= self.initialCoefficients,
            lambdas = self.lambdas,
            reaction = self.computeReaction
        )
    
    def getTimestepResult(self) -> List[Tuple[float, np.array]]:
        if self.timestepResult is None:
            raise "No Timestep Result"
        return self.timestepResult

    # Generates plot for one of the functions
    def generatePlotForFunction(self, functionIndex: int, nPartitionPerAxis: int, stepsize: int = 1, **kwargs) -> None:
        if not (type(stepsize) is int and stepsize > 0):
            raise f"Bad stepsize: {stepsize}"
        timestepped = self.getTimestepResult()

        def computeFunctionValueFromCoeffs(coeffs: np.array, xs: np.array, ys: np.array) -> float:
            xCos: np.array = np.stack([np.array(np.cos(n*xs)) * (1 if n == 0 else 2) for n in range(self.nModes + 1)])
            yCos: np.array = np.stack([np.array(np.cos(m*ys)) * (1 if m == 0 else 2) for m in range(self.nModes + 1)])

            outputMatrix = np.zeros(xs.shape)
            for i in range(xs.shape[0]):
                for j in range(xs.shape[1]):
                    xCosineVector = xCos[:, i, j]
                    yCosineVector = yCos[:, i, j]
                    outputMatrix[i, j] = np.dot(xCosineVector, np.matmul(coeffs, yCosineVector))
            return outputMatrix

        startIndex = functionIndex*(self.nModes + 1)
        endIndex = (functionIndex + 1)*(self.nModes + 1)
        functionHistory = [
            (
                t,
                (lambda coeffs: lambda xs, ys: computeFunctionValueFromCoeffs(coeffs[startIndex:endIndex, :] * self.translatorMatrix, xs, ys))(coefficients)
            )
            for (t, coefficients) in timestepped
        ]
        functionHistory = functionHistory[::stepsize]

        plotXyVectorizedHeatmapMovieOnGrid(
            functionHistory=functionHistory,
            xmin = 0,
            xmax = np.pi,
            ymin = 0,
            ymax = np.pi,
            nPartitionPerAxis=nPartitionPerAxis,
            **kwargs
        )

if __name__ == "__main__":
    class PureDiffusion(GenericExpoHeunTimestepper):
        epsilon: float = None
        def __init__(self, nModes: float, epsilon: float):
            self.epsilon = epsilon
            super().__init__(nModes)

        @override
        def generateLambdas(self) -> np.array:
            return np.array([[
                self.epsilon * (n*n + m*m   )
            for n in range(self.nModes + 1)] for m in range(self.nModes + 1)])
        
        @override
        def computeReaction(self, t: float, currentCoefficients: np.array) -> np.array:
            return currentCoefficients * 0
        
    nModes: int = 20
    epsilon: float = 0.01

    diffusion = PureDiffusion(nModes, epsilon)
    
    # four peaks
    sigma = 10
    initialFunction = lambda x, y: 1 if np.abs(x - np.pi/2) < 1 and np.abs(y - np.pi/2) < 0.5 else 0



    diffusion.initialize([initialFunction])
    print("Initialized.")

    t0: float = 0
    h: float = 0.05
    timesteps: int = 100
    diffusion.runTimestepperAndCache(t0, h, timesteps)
    print("Ran Simulation. Proceeding to plot.")

    diffusion.generatePlotForFunction(
        0,
        100,
        title = None,
        saveAddr  = None,
        show = True,
        interval = 500,
        cmap = 'winter',
        dynamicVs = True
    )


