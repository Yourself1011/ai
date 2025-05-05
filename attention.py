from attentionHead import AttentionHead
from llmlayer import Layer
import numpy as np

from utils import layerNorm


class Attention(Layer):
    def __init__(
        self,
        contextSize: int,
        embedDim: int,
        headCount: int,
        mask: np.typing.NDArray,
    ) -> None:
        self.contextSize = contextSize
        self.embedDim = embedDim
        self.heads = [
            AttentionHead(contextSize, embedDim, headCount, mask)
            for _ in range(headCount)
        ]
        self.g = np.ones((contextSize, embedDim))
        self.b = np.zeros((contextSize, embedDim))

    def feedForward(self, lastLayer: np.typing.NDArray):
        self.a = lastLayer.copy()
        for head in self.heads:
            head.feedForward(lastLayer)
            self.a += head.a

        self.a = layerNorm(self.a, self.g, self.b)
