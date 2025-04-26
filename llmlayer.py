from abc import ABC
import numpy as np


class Layer(ABC):
    def feedForward(self, lastLayer: np.typing.NDArray):
        pass

    def backProp(self, error: np.typing.NDArray):
        pass

    def gradientDescent(self, learningRate: float, batchSize: int):
        pass
