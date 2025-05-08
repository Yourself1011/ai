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
        self.headCount = headCount
        self.heads = [
            AttentionHead(contextSize, embedDim, headCount, mask)
            for _ in range(headCount)
        ]

        self.qkv = np.random.normal(0, 1, (embedDim, embedDim * 3))
        self.proj = np.random.normal(0, 1, (embedDim, embedDim))

        self.g: np.typing.NDArray = np.ones((contextSize, embedDim))
        self.b: np.typing.NDArray = np.zeros((contextSize, embedDim))

    def feedForward(self, lastLayer: np.typing.NDArray):
        q, k, v = [
            np.split(x, self.headCount, axis=1)
            for x in np.split(lastLayer @ self.qkv, 3, axis=1)
        ]
        attentionOutputs = []
        for i in range(len(self.heads)):
            self.heads[i].query = q[i]
            self.heads[i].key = k[i]
            self.heads[i].value = v[i]
            self.heads[i].feedForward(lastLayer)
            attentionOutputs.append(self.heads[i].a)

        self.a = layerNorm(np.hstack(attentionOutputs) @ self.proj, self.g, self.b)
