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
        self.query = q.astype(np.float32)
        self.key = k.astype(np.float32)
        self.value = v.astype(np.float32)

        self.a = softmax(
            np.where(
                self.mask,
                self.query @ np.swapaxes(self.key, -1, -2),
                np.finfo(np.float32).min,
            )
            / (np.sqrt(self.embedDim // self.headCount, dtype=np.float16))
        ).astype(np.float16) @ self.value.astype(np.float16)
        return self.a

    def backProp(self, error: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        weights = softmax(
            np.where(
                self.mask,
                self.query @ np.swapaxes(self.key, -1, -2),
                np.finfo(np.float32).min,
            )
            / (np.sqrt(self.embedDim // self.headCount, dtype=np.float16))
        )

        self.valueError = np.swapaxes(
            weights.astype(np.float16), -1, -2
        ) @ error.astype(np.float16)

        error @= np.swapaxes(self.value, -1, -2)
        sums = (error * weights).sum(-1, keepdims=True)
        error -= sums
        error *= weights / (np.sqrt(self.embedDim // self.headCount))

        self.queryError = error @ self.key
        self.keyError = np.swapaxes(error, -1, -2) @ self.query
        return error, self.valueError

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
