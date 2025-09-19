import time

import numpy as np

try:
    import cupy

    if cupy.cuda.is_available():
        np = cupy
except Exception:
    pass

import numpy.typing as npt

from attentionHead import AttentionHead
from llmlayer import Layer
from utils import layerNorm


class Attention(Layer):
    def __init__(
        self,
        contextSize: int,
        embedDim: int,
        headCount: int,
        mask: npt.NDArray,
        # pool,
    ) -> None:
        self.contextSize = contextSize
        self.embedDim = embedDim
        self.headCount = headCount
        self.heads = [
            AttentionHead(contextSize, embedDim, headCount, mask)
            for _ in range(headCount)
        ]

        self.qkv = np.random.normal(0, 1 / np.sqrt(embedDim), (embedDim, embedDim * 3))
        self.proj = np.random.normal(0, 1 / np.sqrt(embedDim), (embedDim, embedDim))

        self.g: npt.NDArray = np.ones((contextSize, embedDim))
        self.b: npt.NDArray = np.zeros((contextSize, embedDim))

        self.qkvError = np.zeros((embedDim, embedDim * 3))
        self.projError = np.zeros((embedDim, embedDim))

        self.gError: npt.NDArray = np.zeros((contextSize, embedDim))
        self.bError: npt.NDArray = np.zeros((contextSize, embedDim))
        self.error = np.zeros((contextSize, embedDim))
        # self.pool = pool
        super().__init__()

    def feedForward(self, lastLayer: npt.NDArray):
        self.input = lastLayer
        q, k, v = [
            np.split(x, self.headCount, axis=-1)
            for x in np.split(lastLayer @ self.qkv, 3, axis=-1)
        ]
        attentionOutputs = []
        for i in range(len(self.heads)):
            # start = time.time()
            self.heads[i].feedForward(lastLayer, q[i], k[i], v[i])
            # print(time.time() - start)
            attentionOutputs.append(self.heads[i].a)
        # print(self.qkv.max())

        # res = [
        #     self.pool.apply_async(
        #         self.heads[i].feedForward, (lastLayer, q[i], k[i], v[i])
        #     )
        #     for i in range(len(self.heads))
        # ]
        # attentionOutputs = [x.get() for x in res]

        self.combined = np.hstack(attentionOutputs)
        self.a, self.z, self.mean, self.var = layerNorm(
            self.combined @ self.proj, self.g, self.b
        )

    def backProp(self, error: npt.NDArray):
        self.bError += error
        self.gError += error * self.z
        # derivative of layer norm
        error *= self.g
        n = error.shape[-1]
        stdev = np.sqrt(self.var + 1e-5).reshape((-1, 1))
        norm = error * self.z
        sums = norm.sum(-1).reshape((-1, 1))
        errSums = error.sum(-1).reshape((-1, 1))
        error = 1 / (n * stdev) * (n * error - errSums - self.z * sums)

        self.projError += self.combined.T @ error

        splitError = np.split(error @ self.proj.T, self.headCount, axis=-1)
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

        queryError = np.hstack(qkvErrors[0])
        keyError = np.hstack(qkvErrors[1])
        valueError = np.hstack(qkvErrors[2])
        error = np.hstack([queryError, keyError, valueError])
        self.qkvError += self.input.T @ error
        # print(error.shape, self.qkv.shape)
        self.error = error @ self.qkv.T

    def normalizeError(self, batchSize: int):
        self.qkvError /= batchSize
        self.projError /= batchSize

        self.gError /= batchSize
        self.bError /= batchSize
        self.batchSize = batchSize

    def gradientDescent(self, learningRate: float, t: int, mult: float):
        self.b = self.adamW("b", self.b, self.bError, learningRate, t, mult, decay=0)
        self.g = self.adamW("g", self.g, self.gError, learningRate, t, mult, decay=0)
        self.proj = self.adamW("proj", self.proj, self.projError, learningRate, t, mult)
        self.qkv = self.adamW("qkv", self.qkv, self.qkvError, learningRate, t, mult)

        self.qkvError = np.zeros((self.embedDim, self.embedDim * 3))
        self.projError = np.zeros((self.embedDim, self.embedDim))

        self.gError: npt.NDArray = np.zeros((self.contextSize, self.embedDim))
        self.bError: npt.NDArray = np.zeros((self.contextSize, self.embedDim))
        # self.error = np.zeros((self.contextSize, self.embedDim))

        for head in self.heads:
            head.gradientDescent(learningRate, self.batchSize)
