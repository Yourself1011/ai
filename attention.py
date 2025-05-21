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

        self.qkvError = np.zeros((embedDim, embedDim * 3))
        self.projError = np.zeros((embedDim, embedDim))

        self.gError: np.typing.NDArray = np.zeros((contextSize, embedDim))
        self.bError: np.typing.NDArray = np.zeros((contextSize, embedDim))
        self.error = np.zeros((contextSize, embedDim))

    def feedForward(self, lastLayer: np.typing.NDArray):
        self.input = lastLayer
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

        self.combined = np.hstack(attentionOutputs)
        self.a, self.z, self.mean, self.var = layerNorm(
            self.combined @ self.proj, self.g, self.b
        )

    def backProp(self, error: np.typing.NDArray):
        self.bError += error
        self.gError += error * self.z
        # derivative of layer norm
        n = error.shape[-1]
        stdev = np.sqrt(self.var + 1e-5)
        error *= self.g * (1 / (n * stdev)) * (n - 1 - self.z**2)

        self.projError += self.combined.T @ error

        splitError = np.split(error @ self.proj.T, self.headCount, axis=1)
        # print(splitError[0].shape)
        qkvErrors = [[], [], []]
        for i in range(len(self.heads)):
            self.heads[i].backProp(splitError[i])
            qkvErrors[0].append(self.heads[i].queryError)
            qkvErrors[1].append(self.heads[i].keyError)
            qkvErrors[2].append(self.heads[i].valueError)
            # qkvErrors[0].append(
            #     np.zeros((self.contextSize, self.embedDim // self.headCount))
            # )
            # qkvErrors[1].append(
            #     np.zeros((self.contextSize, self.embedDim // self.headCount))
            # )
            # qkvErrors[2].append(
            #     np.zeros((self.contextSize, self.embedDim // self.headCount))
            # )

        error = np.hstack([np.hstack(x) for x in qkvErrors])
        self.qkvError += self.input.T @ error
        # print(error.shape, self.qkv.shape)
        self.error += error @ self.qkv.T

    def gradientDescent(self, learningRate: float, batchSize: int):
        self.b -= self.bError * learningRate / batchSize
        self.g -= self.gError * learningRate / batchSize
        self.proj -= self.projError * learningRate / batchSize
        self.qkv -= self.qkvError * learningRate / batchSize

        self.qkvError = np.zeros((self.embedDim, self.embedDim * 3))
        self.projError = np.zeros((self.embedDim, self.embedDim))

        self.gError: np.typing.NDArray = np.zeros((self.contextSize, self.embedDim))
        self.bError: np.typing.NDArray = np.zeros((self.contextSize, self.embedDim))
        self.error = np.zeros((self.contextSize, self.embedDim))

        # for head in self.heads:
        #     head.gradientDescent(learningRate, batchSize)
