from typing import Tuple

import numpy as np

try:
    import cupy

    if cupy.cuda.is_available():
        np = cupy
except Exception:
    pass
import numpy.typing as npt

from utils import softmax


class AttentionHead:
    def __init__(
        self, contextSize: int, embedDim: int, headCount: int, mask: npt.NDArray
    ) -> None:
        self.contextSize = contextSize
        self.embedDim = embedDim
        self.headCount = headCount
        self.mask = mask
        self.tfMask = mask == False
        # self.query = np.random.normal(0, 1, (embedDim, embedDim // headCount))
        # self.key = np.random.normal(0, 1, (embedDim, embedDim // headCount))
        # self.valueDown = np.random.normal(0, 1, (embedDim, embedDim // headCount))
        # self.valueUp = np.random.normal(0, 1, (embedDim // headCount, embedDim))
        self.query: npt.NDArray = np.empty((contextSize, embedDim // headCount))
        self.key: npt.NDArray = np.empty((contextSize, embedDim // headCount))
        self.value: npt.NDArray = np.empty((contextSize, embedDim // headCount))
        self.a = np.empty((contextSize, embedDim))

        self.queryError: npt.NDArray = np.empty((contextSize, embedDim // headCount))
        self.keyError: npt.NDArray = np.empty((contextSize, embedDim // headCount))
        self.valueError: npt.NDArray = np.empty((contextSize, embedDim // headCount))

    def feedForward(
        self,
        lastLayer: npt.NDArray,
        q: npt.NDArray,
        k: npt.NDArray,
        v: npt.NDArray,
    ):
        self.query = q
        self.key = k
        self.value = v
        attentionPattern = np.where(
            self.mask,
            self.query @ self.key.T,
            -np.inf,
        ) / (np.sqrt(self.embedDim // self.headCount))
        # attentionPattern = self.query @ self.key.T / (np.sqrt(self.embedDim // self.headCount))

        self.weights = softmax(attentionPattern)
        # value = np.matmul(lastLayer, np.matmul(self.valueUp, self.valueDown))
        # print("alskdjf")
        # change = (
        #     value.reshape(1, contextSize, self.embedDim)
        #     * weights.reshape(contextSize, contextSize, 1)
        # ).sum(1)

        self.a = self.weights @ self.value
        return self.a

    def backProp(
        self, error: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        self.valueError = self.weights.T @ error

        error = error @ self.value.T
        sums = (error * self.weights).sum(-1).reshape((-1, 1))
        error = self.weights * (error - sums) / np.sqrt(self.embedDim)

        self.queryError = error @ self.key
        self.keyError = error.T @ self.query
        return self.queryError, self.keyError, self.valueError

    def gradientDescent(self, learningRate: float, batchSize: int):
        self.queryError: npt.NDArray = np.empty(
            (self.contextSize, self.embedDim // self.headCount)
        )
        self.keyError: npt.NDArray = np.empty(
            (self.contextSize, self.embedDim // self.headCount)
        )
        self.valueError: npt.NDArray = np.empty(
            (self.contextSize, self.embedDim // self.headCount)
        )
