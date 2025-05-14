from llmlayer import Layer
import numpy as np

from utils import gelu, layerNorm, geluCoefficient


class Mlp(Layer):
    def __init__(
        self,
        contextSize: int,
        embedDim: int,
    ) -> None:
        self.contextSize = contextSize
        self.embedDim = embedDim
        self.w: list[np.typing.NDArray] = [
            np.random.normal(0, 1, (embedDim, 4 * embedDim)),
            np.random.normal(0, 1, (4 * embedDim, embedDim)),
        ]
        self.b: list[np.typing.NDArray] = [
            np.zeros(4 * embedDim),
            np.zeros(embedDim),
        ]
        self.g: np.typing.NDArray = np.ones((contextSize, embedDim))
        self.beta: np.typing.NDArray = np.zeros((contextSize, embedDim))

        self.wError: list[np.typing.NDArray] = [
            np.zeros((embedDim, 4 * embedDim)),
            np.zeros((4 * embedDim, embedDim)),
        ]
        self.bError: list[np.typing.NDArray] = [
            np.zeros(4 * embedDim),
            np.zeros(embedDim),
        ]
        self.gError: np.typing.NDArray = np.zeros((contextSize, embedDim))
        self.betaError: np.typing.NDArray = np.zeros((contextSize, embedDim))
        self.error = np.zeros((contextSize, embedDim))

    def feedForward(self, lastLayer: np.typing.NDArray):
        self.input = lastLayer
        self.layer1 = lastLayer @ self.w[0] + self.b[0]
        self.gelu, self.tanh, self.inside = gelu(self.layer1)
        self.layer2 = self.gelu @ self.w[1] + self.b[1]
        self.a, self.z, self.mean, self.var = layerNorm(self.layer2, self.g, self.beta)

    def backProp(self, error: np.typing.NDArray):
        self.betaError += error
        self.gError += error * self.z
        # derivative of layer norm
        n = error.shape[-1]
        stdev = np.sqrt(self.var + 1e-5)
        error *= self.g * (
            (n - 1) / (n * stdev) - (self.layer2 - self.mean) ** 2 / (n * stdev**3)
        )
        # print(error.shape)
        self.bError[1] += error.sum(0)
        # print(self.gelu.shape)
        self.wError[1] += self.gelu.T @ error
        error = (
            (error @ self.w[1].T)
            * 0.5
            * (
                (1 + self.tanh)
                + self.layer1
                / np.cosh(self.inside) ** 2
                * geluCoefficient
                * (1 + 0.134145 * self.layer1**2)
            )
        )
        # print(error)
        self.bError[0] += error.sum(0)
        # print(self.input.shape, error.shape)
        self.wError[0] += self.input.T @ error
        self.error += error @ self.w[0].T

    def gradientDescent(self, learningRate: float, batchSize: int):
        self.beta -= self.betaError * learningRate / batchSize
        self.g -= self.gError * learningRate / batchSize
        self.b[1] -= self.bError[1] * learningRate / batchSize
        self.w[1] -= self.wError[1] * learningRate / batchSize
        self.b[0] -= self.bError[0] * learningRate / batchSize
        self.w[0] -= self.wError[0] * learningRate / batchSize
        self.wError: list[np.typing.NDArray] = [
            np.zeros((self.embedDim, 4 * self.embedDim)),
            np.zeros((4 * self.embedDim, self.embedDim)),
        ]
        self.bError: list[np.typing.NDArray] = [
            np.zeros(4 * self.embedDim),
            np.zeros(self.embedDim),
        ]
        self.gError: np.typing.NDArray = np.zeros((self.contextSize, self.embedDim))
        self.betaError: np.typing.NDArray = np.zeros((self.contextSize, self.embedDim))
        self.error = np.zeros((self.contextSize, self.embedDim))
