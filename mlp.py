from llmlayer import Layer
import numpy as np

from utils import gelu, layerNorm


class Mlp(Layer):
    def __init__(
        self,
        contextSize: int,
        embedDim: int,
    ) -> None:
        self.contextSize = contextSize
        self.embedDim = embedDim
        self.w: tuple[np.typing.NDArray, np.typing.NDArray] = (
            np.random.normal(0, 1, (embedDim, 4 * embedDim)),
            np.random.normal(0, 1, (4 * embedDim, embedDim)),
        )
        self.b: tuple[np.typing.NDArray, np.typing.NDArray] = (
            np.zeros(4 * embedDim),
            np.zeros(embedDim),
        )
        self.g: np.typing.NDArray = np.ones((contextSize, embedDim))
        self.beta: np.typing.NDArray = np.zeros((contextSize, embedDim))

        self.wError: tuple[np.typing.NDArray, np.typing.NDArray] = (
            np.zeros((embedDim, 4 * embedDim)),
            np.zeros((4 * embedDim, embedDim)),
        )
        self.bError: tuple[np.typing.NDArray, np.typing.NDArray] = (
            np.zeros(4 * embedDim),
            np.zeros(embedDim),
        )
        self.gError: np.typing.NDArray = np.zeros((contextSize, embedDim))
        self.betaError: np.typing.NDArray = np.zeros((contextSize, embedDim))

    def feedForward(self, lastLayer: np.typing.NDArray):
        self.a = lastLayer @ self.w[0] + self.b[0]
        self.a = gelu(self.a)
        self.a = self.a @ self.w[1] + self.b[1]
        self.a, self.z = layerNorm(self.a, self.g, self.beta)

    def backProp(self, error: np.typing.NDArray):
        self.betaError += error
        # self.gError += error * self.z

    def gradientDescent(self, learningRate: float, batchSize: int):
        self.beta -= self.betaError * learningRate / batchSize
        self.g -= self.gError * learningRate / batchSize
        self.wError: tuple[np.typing.NDArray, np.typing.NDArray] = (
            np.zeros((self.embedDim, 4 * self.embedDim)),
            np.zeros((4 * self.embedDim, self.embedDim)),
        )
        self.bError: tuple[np.typing.NDArray, np.typing.NDArray] = (
            np.zeros(4 * self.embedDim),
            np.zeros(self.embedDim),
        )
        self.gError: np.typing.NDArray = np.zeros((self.contextSize, self.embedDim))
        self.betaError: np.typing.NDArray = np.zeros((self.contextSize, self.embedDim))
