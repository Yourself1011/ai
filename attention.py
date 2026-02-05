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

        self.qkv = np.random.normal(
            0, 1 / np.sqrt(embedDim), (embedDim, embedDim * 3)
        ).astype(np.float32)
        self.proj = np.random.normal(
            0, 1 / np.sqrt(embedDim), (embedDim, embedDim)
        ).astype(np.float32)

        self.g: npt.NDArray = np.ones((contextSize, embedDim), dtype=np.float32)
        self.b: npt.NDArray = np.zeros((contextSize, embedDim), dtype=np.float32)

        self.qkv16 = self.qkv.astype(np.float16)
        self.proj16 = self.proj.astype(np.float16)
        self.g16 = self.g.astype(np.float16)
        self.b16 = self.b.astype(np.float16)

        self.qkvError = np.zeros((embedDim, embedDim * 3), dtype=np.float32)
        self.projError = np.zeros((embedDim, embedDim), dtype=np.float32)

        self.gError: npt.NDArray = np.zeros((contextSize, embedDim), dtype=np.float32)
        self.bError: npt.NDArray = np.zeros((contextSize, embedDim), dtype=np.float32)
        self.error = np.zeros((contextSize, embedDim), dtype=np.float32)
        # self.pool = pool
        super().__init__()

    def feedForward(self, lastLayer: npt.NDArray):
        self.input = lastLayer
        self.lnOut, self.z, self.mean, self.var = layerNorm(
            lastLayer.astype(np.float32), self.g16, self.b16
        )

        self.q, self.k, self.v = np.split(self.lnOut @ self.qkv16, 3, axis=-1)

        qHead = np.split(self.q, self.headCount, axis=-1)
        kHead = np.split(self.k, self.headCount, axis=-1)
        vHead = np.split(self.v, self.headCount, axis=-1)

        attentionOutputs = []
        for i in range(len(self.heads)):
            # start = time.time()
            self.heads[i].feedForward(self.lnOut, qHead[i], kHead[i], vHead[i])
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

        self.combined = np.concatenate(attentionOutputs, axis=-1)
        self.a = self.combined @ self.proj16 + self.input

    def backProp(self, error: npt.NDArray):
        initError = error
        self.projError += (
            np.swapaxes(self.combined, -1, -2) @ error.astype(np.float16)
        ).sum(0)

        splitError = np.split(
            error @ np.swapaxes(self.proj, -1, -2), self.headCount, axis=-1
        )
        # print(splitError[0].shape)
        qkErrors = []
        valueErrors = []
        for i in range(len(self.heads)):
            qkError = self.heads[i].backProp(splitError[i])[0]
            qkErrors.append(qkError)
            valueErrors.append(self.heads[i].valueError)
            # qkvErrors[0].append(
            #     np.zeros((self.contextSize, self.embedDim // self.headCount))
            # )
            # qkvErrors[1].append(
            #     np.zeros((self.contextSize, self.embedDim // self.headCount))
            # )
            # qkvErrors[2].append(
            #     np.zeros((self.contextSize, self.embedDim // self.headCount))
            # )

        queryError = np.concatenate(qkErrors @ self.k, axis=-1)
        keyError = np.concatenate(np.swapaxes(qkErrors, -1, -2) @ self.q, axis=-1)
        valueError = np.concatenate(valueErrors, axis=-1)

        error = np.concatenate([queryError, keyError, valueError], axis=-1)
        self.qkvError += (np.swapaxes(self.lnOut, -1, -2) @ error).sum(0)
        # print(error.shape, self.qkv.shape)
        error = error @ np.swapaxes(self.qkv, -1, -2)

        self.bError += error.astype(np.float16).sum(0)
        self.gError += (error.astype(np.float16) * self.z).sum(0)
        # derivative of layer norm
        error *= self.g
        n = error.shape[-1]
        stdev = np.sqrt(self.var + 1e-5)
        norm = error * self.z
        sums = norm.sum(-1, keepdims=True)
        errSums = error.sum(-1, keepdims=True)
        self.error = 1 / (n * stdev) * (n * error - errSums - self.z * sums) + initError

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

        self.qkv16 = self.qkv.astype(np.float16)
        self.proj16 = self.proj.astype(np.float16)
        self.g16 = self.g.astype(np.float16)
        self.b16 = self.b.astype(np.float16)

        self.qkvError = np.zeros((self.embedDim, self.embedDim * 3))
        self.projError = np.zeros((self.embedDim, self.embedDim))

        self.gError: npt.NDArray = np.zeros((self.contextSize, self.embedDim))
        self.bError: npt.NDArray = np.zeros((self.contextSize, self.embedDim))
        # self.error = np.zeros((self.contextSize, self.embedDim))

        for head in self.heads:
            head.gradientDescent(learningRate, self.batchSize)
