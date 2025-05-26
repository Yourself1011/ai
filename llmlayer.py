import numpy as np


class LLMBase:
    def __init__(self):
        self.m = {}
        self.v = {}

    def adamW(
        self,
        name: str,
        value: np.typing.NDArray,
        change: np.typing.NDArray,
        lr: float,
        t: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        decay: float = 1e-2,
    ):
        if name not in self.m:
            self.m[name] = np.zeros(value.shape)
            self.v[name] = np.zeros(value.shape)

        self.m[name] = self.m[name] * beta1 + change * (1 - beta2)
        self.v[name] = self.v[name] * beta2 + change**2 * (1 - beta2)
        m = self.m[name] / (1 - beta1**t)
        v = self.v[name] / (1 - beta2**t)

        # return change * lr
        return lr * (m / (np.sqrt(v) + 1e-8) + decay * value)


class Layer(LLMBase):
    def __init__(self):
        super().__init__()

    def feedForward(self, lastLayer: np.typing.NDArray):
        pass

    def backProp(self, error: np.typing.NDArray):
        pass

    def gradientDescent(self, learningRate: float, batchSize: int, t: int):
        pass
