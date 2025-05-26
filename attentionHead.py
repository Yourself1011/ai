import math
from typing import Tuple
from llmlayer import Layer
import numpy as np

from utils import softmax


class AttentionHead:
    def __init__(
        self, contextSize: int, embedDim: int, headCount: int, mask: np.typing.NDArray
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
        self.query: np.typing.NDArray = np.empty((contextSize, embedDim // headCount))
        self.key: np.typing.NDArray = np.empty((contextSize, embedDim // headCount))
        self.value: np.typing.NDArray = np.empty((contextSize, embedDim // headCount))
        self.a = np.empty((contextSize, embedDim))

        self.queryError: np.typing.NDArray = np.empty(
            (contextSize, embedDim // headCount)
        )
        self.keyError: np.typing.NDArray = np.empty(
            (contextSize, embedDim // headCount)
        )
        self.valueError: np.typing.NDArray = np.empty(
            (contextSize, embedDim // headCount)
        )

    def feedForward(
        self,
        lastLayer: np.typing.NDArray,
        q: np.typing.NDArray,
        k: np.typing.NDArray,
        v: np.typing.NDArray,
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
        self, error: np.typing.NDArray
    ) -> Tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
        self.valueError = self.weights.T @ error
        error = np.where(
            self.tfMask,
            (error @ self.value.T) * self.weights * (1 - self.weights),
            0,
        ) / (np.sqrt(self.embedDim))
        self.queryError = error @ self.key
        self.keyError = (self.query.T @ error).T
        return self.queryError, self.keyError, self.valueError

    def gradientDescent(self, learningRate: float, batchSize: int, t):
        self.queryError: np.typing.NDArray = np.empty(
            (self.contextSize, self.embedDim // self.headCount)
        )
        self.keyError: np.typing.NDArray = np.empty(
            (self.contextSize, self.embedDim // self.headCount)
        )
        self.valueError: np.typing.NDArray = np.empty(
            (self.contextSize, self.embedDim // self.headCount)
        )
