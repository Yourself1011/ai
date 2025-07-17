import numpy as np

from llmlayer import Layer

try:
    import cupy

    if cupy.cuda.is_available():
        np = cupy
except Exception:
    pass
import time

import numpy.typing as npt

from utils import layerNorm, sigmoid


class Mlp(Layer):
    def __init__(
        self,
        contextSize: int,
        embedDim: int,
    ) -> None:
        self.contextSize = contextSize
        self.embedDim = embedDim
        self.w: list[npt.NDArray] = [
            np.random.normal(0, 1, (embedDim, 4 * embedDim)),
            np.random.normal(0, 1, (4 * embedDim, embedDim)),
        ]
        self.b: list[npt.NDArray] = [
            np.zeros(4 * embedDim),
            np.zeros(embedDim),
        ]
        self.g: npt.NDArray = np.ones((contextSize, embedDim))
        self.beta: npt.NDArray = np.zeros((contextSize, embedDim))

        self.wError: list[npt.NDArray] = [
            np.zeros((embedDim, 4 * embedDim)),
            np.zeros((4 * embedDim, embedDim)),
        ]
        self.bError: list[npt.NDArray] = [
            np.zeros(4 * embedDim),
            np.zeros(embedDim),
        ]
        self.gError: npt.NDArray = np.zeros((contextSize, embedDim))
        self.betaError: npt.NDArray = np.zeros((contextSize, embedDim))
        self.error = np.zeros((contextSize, embedDim))
        super().__init__()

    def feedForward(self, lastLayer: npt.NDArray):
        self.input = lastLayer
        # start = time.time()
        self.layer1 = lastLayer @ self.w[0] + self.b[0]
        # self.gelu, self.tanh, self.inside = gelu(self.layer1)
        # use sigmoid approximation instad of tanh
        self.multiplied = self.layer1 * 1.702
        self.sigmoid = sigmoid(self.multiplied)
        self.gelu = self.layer1 * self.sigmoid
        self.layer2 = self.gelu @ self.w[1] + self.b[1]
        self.a, self.z, self.mean, self.var = layerNorm(self.layer2, self.g, self.beta)
        # print(time.time() - start)

    def backProp(self, error: npt.NDArray):
        self.betaError += error
        self.gError += error * self.z
        # derivative of layer norm
        error *= self.g
        n = error.shape[-1]
        stdev = np.sqrt(self.var + 1e-5).reshape((-1, 1))
        norm = error * self.z
        sums = norm.sum(-1).reshape((-1, 1))
        errSums = error.sum(-1).reshape((-1, 1))
        error = 1 / (n * stdev) * (n * error - errSums - self.z * sums)
        # print(error.shape)
        self.bError[1] += error.sum(0)
        # print((self.gelu.T @ error).sum())
        # print(self.gelu.shape, error.shape)
        self.wError[1] += self.gelu.T @ error
        error = self.sigmoid * (
            1 + self.multiplied * (1 - self.sigmoid)) * (error @ self.w[1].T)
        
        # print(error)
        self.bError[0] += error.sum(0)
        # print(self.input.shape, error.shape)
        self.wError[0] += self.input.T @ error
        # print(error.shape, self.w[0].shape)
        self.error = error @ self.w[0].T
        # self.error += ( self.w[0] @ error.T ).T

    def normalizeError(self, batchSize: int):
        self.wError[0] /= batchSize
        self.wError[1] /= batchSize
        self.bError[0] /= batchSize
        self.bError[1] /= batchSize

        self.gError /= batchSize
        self.betaError /= batchSize
        self.error /= batchSize

    def gradientDescent(self, learningRate: float, t: int, mult: float):
        self.beta -= self.adamW(
            "beta", self.beta, self.betaError, learningRate, t, mult, decay=0
        )
        self.g -= self.adamW("g", self.g, self.gError, learningRate, t, mult, decay=0)
        self.b[1] -= self.adamW(
           "b1", self.b[1], self.bError[1], learningRate, t, mult, decay=0
        )
        self.w[1] -= self.adamW("w1", self.w[1], self.wError[1], learningRate, t, mult)
        self.b[0] -= self.adamW(
            "b0", self.b[0], self.bError[0], learningRate, t, mult, decay=0
        )
        self.w[0] -= self.adamW("w0", self.w[0], self.wError[0], learningRate, t, mult)

        self.wError: list[npt.NDArray] = [
            np.zeros((self.embedDim, 4 * self.embedDim)),
            np.zeros((4 * self.embedDim, self.embedDim)),
        ]
        self.bError: list[npt.NDArray] = [
            np.zeros(4 * self.embedDim),
            np.zeros(self.embedDim),
        ]
        self.gError: npt.NDArray = np.zeros((self.contextSize, self.embedDim))
        self.betaError: npt.NDArray = np.zeros((self.contextSize, self.embedDim))
        # self.error = np.zeros((self.contextSize, self.embedDim))
