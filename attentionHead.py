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
        # self.tfMask = mask == False
        # self.query = np.random.normal(0, 1, (embedDim, embedDim // headCount))
        # self.key = np.random.normal(0, 1, (embedDim, embedDim // headCount))
        # self.valueDown = np.random.normal(0, 1, (embedDim, embedDim // headCount))
        # self.valueUp = np.random.normal(0, 1, (embedDim // headCount, embedDim))
        self.query: npt.NDArray = np.empty(
            (contextSize, embedDim // headCount), dtype=np.float16
        )
        self.key: npt.NDArray = np.empty(
            (contextSize, embedDim // headCount), dtype=np.float16
        )
        self.value: npt.NDArray = np.empty(
            (contextSize, embedDim // headCount), dtype=np.float16
        )
        self.a = np.empty((contextSize, embedDim), dtype=np.float16)

        self.queryError: npt.NDArray = np.empty(
            (contextSize, embedDim // headCount), dtype=np.float32
        )
        self.keyError: npt.NDArray = np.empty(
            (contextSize, embedDim // headCount), dtype=np.float32
        )
        self.valueError: npt.NDArray = np.empty(
            (contextSize, embedDim // headCount), dtype=np.float32
        )

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
            self.query @ np.swapaxes(self.key, -1, -2),
            -65500,
        ) / (np.sqrt(self.embedDim // self.headCount, dtype=np.float16))
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
        self.valueError = np.swapaxes(self.weights, -1, -2) @ error

        error = error @ np.swapaxes(self.value, -1, -2)
        sums = (error * self.weights).sum(-1, keepdims=True)
        error = (
            self.weights * (error - sums) / (np.sqrt(self.embedDim // self.headCount))
        )

        self.queryError = error @ self.key
        self.keyError = np.swapaxes(error, -1, -2) @ self.query
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
