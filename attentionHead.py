import math
from typing import Tuple

import torch as np

try:
    import cupy

    if cupy.cuda.is_available():
        np = cupy
except Exception:
    pass
import torch.types as npt

from utils import softmax


class AttentionHead:
    def __init__(
        self, contextSize: int, embedDim: int, headCount: int, mask: npt.Tensor
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
        self.query = np.empty((contextSize, embedDim // headCount), requires_grad=True)
        self.key = np.empty((contextSize, embedDim // headCount), requires_grad=True)
        self.value = np.empty((contextSize, embedDim // headCount), requires_grad=True)
        self.a = np.empty((contextSize, embedDim), requires_grad=True)

        self.queryError = np.empty((contextSize, embedDim // headCount))
        self.keyError = np.empty((contextSize, embedDim // headCount))
        self.valueError = np.empty((contextSize, embedDim // headCount))

    def feedForward(
        self,
        lastLayer: npt.Tensor,
        q: npt.Tensor,
        k: npt.Tensor,
        v: npt.Tensor,
    ):
        self.query = q
        self.key = k
        self.value = v
        attentionPattern = np.where(
            self.mask,
            self.query @ self.key.T,
            -1e9,
        ) / (math.sqrt(self.embedDim // self.headCount))
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

    def backProp(self, error: npt.Tensor) -> Tuple[npt.Tensor, npt.Tensor, npt.Tensor]:
        self.valueError = self.weights.T @ error

        error = error @ self.value.T
        sums = (error * self.weights).sum(-1).reshape((-1, 1))
        error = (
            self.weights * (error - sums) / (math.sqrt(self.embedDim // self.headCount))
        )

        self.queryError = error @ self.key
        self.keyError = error.T @ self.query
        return self.queryError, self.keyError, self.valueError

    def gradientDescent(self, learningRate: float, batchSize: int):
        self.queryError = np.empty((self.contextSize, self.embedDim // self.headCount))
        self.keyError = np.empty((self.contextSize, self.embedDim // self.headCount))
        self.valueError = np.empty((self.contextSize, self.embedDim // self.headCount))
