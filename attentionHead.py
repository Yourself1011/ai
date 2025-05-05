import math
from llmlayer import Layer
import numpy as np

from utils import softmax


class AttentionHead(Layer):
    def __init__(
        self, contextSize: int, embedDim: int, headCount: int, mask: np.typing.NDArray
    ) -> None:
        self.embedDim = embedDim
        self.headCount = headCount
        self.mask = mask
        self.query = np.random.normal(0, 1, (embedDim, embedDim // headCount))
        self.key = np.random.normal(0, 1, (embedDim, embedDim // headCount))
        self.valueDown = np.random.normal(0, 1, (embedDim, embedDim // headCount))
        self.valueUp = np.random.normal(0, 1, (embedDim // headCount, embedDim))
        self.a = np.empty((contextSize, embedDim))

    def feedForward(self, lastLayer: np.typing.NDArray):
        attentionPattern = np.where(
            self.mask,
            # (
            #     np.matmul(lastLayer, self.query)
            #     * np.matmul(lastLayer, self.key).reshape(
            #         contextSize, 1, self.embedDim // self.headCount
            #     )
            # ).sum(2),
            (lastLayer @ self.query) @ (lastLayer @ self.key).T,
            -math.inf,
        ) / (np.sqrt(self.embedDim))

        weights = softmax(attentionPattern)
        # value = np.matmul(lastLayer, np.matmul(self.valueUp, self.valueDown))
        # print("alskdjf")
        # change = (
        #     value.reshape(1, contextSize, self.embedDim)
        #     * weights.reshape(contextSize, contextSize, 1)
        # ).sum(1)

        self.a = (weights @ (lastLayer @ self.valueDown)) @ self.valueUp
