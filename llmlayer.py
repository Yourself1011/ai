import numpy as np
import numpy.typing as npt


class LLMBase:
    def __init__(self):
        self.m = {}
        self.v = {}

    def adamW(
        self,
        name: str,
        value: npt.NDArray,
        change: npt.NDArray,
        lr: float,
        t: int,
        mult: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        decay: float = 0.01,
    ):
        if name not in self.m:
            self.m[name] = np.zeros(value.shape)
            self.v[name] = np.zeros(value.shape)

        if mult != 1:
            change *= mult

        self.m[name] = self.m[name] * beta1 + change * (1 - beta2)
        self.v[name] = self.v[name] * beta2 + change**2 * (1 - beta2)
        m = self.m[name] / (1 - beta1**t)
        v = self.v[name] / (1 - beta2**t)

        if decay == 0:
            res = m / (np.sqrt(v) + 1e-8)
        else:
            res = m / (np.sqrt(v) + 1e-8) + decay * value

        # return change * lr
        return lr * res


class Layer(LLMBase):
    def __init__(self):
        super().__init__()

    def feedForward(self, lastLayer: npt.NDArray):
        pass

    def backProp(self, error: npt.NDArray):
        pass

    def normalizeError(self, batchSize: int):
        pass

    def gradientDescent(self, learningRate: float, t: int, mult: float):
        pass
