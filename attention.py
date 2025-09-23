import time

import torch as np

try:
    import cupy

    if cupy.cuda.is_available():
        np = cupy
except Exception:
    pass

import torch.types as npt


from attentionHead import AttentionHead
from llmlayer import Layer
from utils import layerNorm


class Attention(Layer):
    def __init__(
        self,
        contextSize: int,
        embedDim: int,
        headCount: int,
        mask: npt.Tensor,
        # pool,
    ) -> None:
        self.contextSize = contextSize
        self.embedDim = embedDim
        self.headCount = headCount
        self.heads = [
            AttentionHead(contextSize, embedDim, headCount, mask)
            for _ in range(headCount)
        ]

        self.qkv = np.normal(0, 0.02, (embedDim, embedDim * 3), requires_grad=True)
        self.proj = np.normal(0, 0.02, (embedDim, embedDim), requires_grad=True)

        self.g = np.ones((contextSize, embedDim), requires_grad=True)
        self.b = np.zeros((contextSize, embedDim), requires_grad=True)

        self.qkvError = np.zeros((embedDim, embedDim * 3))
        self.projError = np.zeros((embedDim, embedDim))

        self.gError = np.zeros((contextSize, embedDim))
        self.bError = np.zeros((contextSize, embedDim))
        self.error = np.zeros((contextSize, embedDim))
        # self.pool = pool
        super().__init__()

    def feedForward(self, lastLayer):
        self.input = lastLayer
        lastLayer, self.z, self.mean, self.var = layerNorm(lastLayer, self.g, self.b)

        q, k, v = [
            np.split(x, self.embedDim // self.headCount, dim=-1)
            for x in np.split(lastLayer @ self.qkv, self.embedDim, dim=-1)
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
        self.a = self.combined @ self.proj + self.input

    def backProp(self, error):
        initError = error
        self.projError += self.combined.T @ error

        splitError = np.split(
            error @ self.proj.T, self.embedDim // self.headCount, dim=-1
        )
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
        error = error @ self.qkv.T
        self.bError += error
        self.gError += error * self.z
        # derivative of layer norm
        error *= self.g
        n = error.shape[-1]
        stdev = np.sqrt(self.var + 1e-5).reshape((-1, 1))
        norm = error * self.z
        sums = norm.sum(-1).reshape((-1, 1))
        errSums = error.sum(-1).reshape((-1, 1))
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

        self.qkvError = np.zeros((self.embedDim, self.embedDim * 3))
        self.projError = np.zeros((self.embedDim, self.embedDim))

        self.gError = np.zeros((self.contextSize, self.embedDim))
        self.bError = np.zeros((self.contextSize, self.embedDim))
        # self.error = np.zeros((self.contextSize, self.embedDim))

        for head in self.heads:
            head.gradientDescent(learningRate, self.batchSize)
