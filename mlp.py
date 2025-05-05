from llmlayer import Layer
import numpy as np

from utils import gelu, layerNorm


class Mlp(Layer):
    def __init__(
        self,
        contextSize: int,
        embedDim: int,
    ) -> None:
        self.w = (
            np.random.normal(0, 1, (embedDim, 4 * embedDim)),
            np.random.normal(0, 1, (4 * embedDim, embedDim)),
        )
        self.b = (np.zeros(4 * embedDim), np.zeros(embedDim))
        self.g = np.ones((contextSize, embedDim))
        self.beta = np.zeros((contextSize, embedDim))

    def feedForward(self, lastLayer: np.typing.NDArray):
        self.a = lastLayer @ self.w[0] + self.b[0]
        self.a = gelu(self.a)
        self.a = self.a @ self.w[1] + self.b[1]
        self.a = layerNorm(self.a, self.g, self.beta)
