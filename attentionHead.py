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
        self.tfMask = mask == 0
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
            # (
            #     np.matmul(lastLayer, self.query)
            #     * np.matmul(lastLayer, self.key).reshape(
            #         contextSize, 1, self.embedDim // self.headCount
            #     )
            # ).sum(2),
            self.query @ self.key.T,
            -np.inf,
        ) / (np.sqrt(self.embedDim))

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
        error = np.where(
            self.tfMask,
            (error @ self.value.T)
            * self.weights
            * (1 - self.weights)
            / (np.sqrt(self.embedDim)),
            0,
        )
        self.queryError = error @ self.key
        self.keyError = (self.query.T @ error).T
        return self.queryError, self.keyError, self.valueError

    def gradientDescent(self, learningRate: float, batchSize: int, t):
        self.queryError: npt.NDArray = np.empty(
            (self.contextSize, self.embedDim // self.headCount)
        )
        self.keyError: npt.NDArray = np.empty(
            (self.contextSize, self.embedDim // self.headCount)
        )
        self.valueError: npt.NDArray = np.empty(
            (self.contextSize, self.embedDim // self.headCount)
        )
