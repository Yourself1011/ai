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
            np.random.normal(0, 1 / np.sqrt(embedDim), (embedDim, 4 * embedDim)).astype(
                np.float32
            ),
            np.random.normal(0, 1 / np.sqrt(embedDim), (4 * embedDim, embedDim)).astype(
                np.float32
            ),
        ]
        self.b: list[npt.NDArray] = [
            np.zeros(4 * embedDim, dtype=np.float32),
            np.zeros(embedDim, dtype=np.float32),
        ]
        self.g: npt.NDArray = np.ones((contextSize, embedDim), dtype=np.float32)
        self.beta: npt.NDArray = np.zeros((contextSize, embedDim), dtype=np.float32)

        self.wError: list[npt.NDArray] = [
            np.zeros((embedDim, 4 * embedDim), dtype=np.float32),
            np.zeros((4 * embedDim, embedDim), dtype=np.float32),
        ]
        self.bError: list[npt.NDArray] = [
            np.zeros(4 * embedDim, dtype=np.float32),
            np.zeros(embedDim, dtype=np.float32),
        ]
        self.gError: npt.NDArray = np.zeros((contextSize, embedDim), dtype=np.float32)
        self.betaError: npt.NDArray = np.zeros(
            (contextSize, embedDim), dtype=np.float32
        )
        self.error = np.zeros((contextSize, embedDim), dtype=np.float32)
        super().__init__()

    def feedForward(self, lastLayer: npt.NDArray):
        self.input = lastLayer
        self.lnOut, self.z, self.mean, self.var = layerNorm(
            lastLayer, self.g, self.beta
        )
        lastLayer = self.lnOut
        # start = time.time()
        self.layer1 = lastLayer @ self.w[0] + self.b[0]
        # self.gelu, self.tanh, self.inside = gelu(self.layer1)
        # use sigmoid approximation instad of tanh
        self.multiplied = self.layer1 * 1.702
        self.sigmoid = sigmoid(self.multiplied)
        self.gelu = self.layer1 * self.sigmoid
        self.layer2 = self.gelu @ self.w[1] + self.b[1]
        self.a = self.layer2 + self.input
        # print(time.time() - start)

    def backProp(self, error: npt.NDArray):
        initError = error
        # print(error.shape)
        self.bError[1] += error.sum(0).sum(0)
        # print((self.gelu.T @ error).sum())
        # print(self.gelu.shape, error.shape)
        self.wError[1] += (np.swapaxes(self.gelu, -1, -2) @ error).sum(0)
        error = (
            self.sigmoid
            * (1 + self.multiplied * (1 - self.sigmoid))
            * (error @ np.swapaxes(self.w[1], -1, -2))
        )

        # print(error)
        self.bError[0] += error.sum(0).sum(0)
        # print(self.input.shape, error.shape)
        self.wError[0] += (np.swapaxes(self.lnOut, -1, -2) @ error).sum(0)
        # print(error.shape, self.w[0].shape)
        error = error @ np.swapaxes(self.w[0], -1, -2)
        # self.error += ( self.w[0] @ error.T ).T
        self.betaError += error.sum(0)
        self.gError += (error * self.z).sum(0)

        # derivative of layer norm
        error *= self.g
        n = error.shape[-1]
        stdev = np.sqrt(self.var + 1e-5)
        norm = error * self.z
        sums = norm.sum(-1, keepdims=True)
        errSums = error.sum(-1, keepdims=True)
        self.error = 1 / (n * stdev) * (n * error - errSums - self.z * sums) + initError

    def normalizeError(self, batchSize: int):
        self.wError[0] /= batchSize
        self.wError[1] /= batchSize
        self.bError[0] /= batchSize
        self.bError[1] /= batchSize

        self.gError /= batchSize
        self.betaError /= batchSize
        self.error /= batchSize

    def gradientDescent(self, learningRate: float, t: int, mult: float):
        self.beta = self.adamW(
            "beta", self.beta, self.betaError, learningRate, t, mult, decay=0
        )
        self.g = self.adamW("g", self.g, self.gError, learningRate, t, mult, decay=0)
        self.b[1] = self.adamW(
            "b1", self.b[1], self.bError[1], learningRate, t, mult, decay=0
        )
        self.w[1] = self.adamW("w1", self.w[1], self.wError[1], learningRate, t, mult)
        self.b[0] = self.adamW(
            "b0", self.b[0], self.bError[0], learningRate, t, mult, decay=0
        )
        self.w[0] = self.adamW("w0", self.w[0], self.wError[0], learningRate, t, mult)

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
