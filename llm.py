import random
import time
from typing import Dict
from attention import Attention
from attentionHead import AttentionHead
from embedding import Embedding
from mlp import Mlp
from tokenizer import decode, encode, load
import numpy as np
from utils import layerNorm, softmax


# gpt-2 124m hyperparams
# context size: 1024
# vocab size: 50257
# layers: 12
# heads: 12
# embed dimension: 768
class LLM:
    def __init__(
        self,
        vocabSize: int,
        embedDim: int,
        contextSize: int,
        headCount: int,
        layerCount: int,
    ):
        self.vocabSize = vocabSize
        self.embedDim = embedDim
        self.contextSize = contextSize
        self.headCount = headCount
        self.layerCount = layerCount
        (self.merges, self.vocab) = load(vocabSize)

        self.embedding = Embedding(vocabSize, embedDim, contextSize)
        self.g = np.ones((contextSize, embedDim))
        self.b = np.zeros((contextSize, embedDim))

        attentionMask = np.full((self.contextSize, self.contextSize), False)
        for i in range(self.contextSize):
            attentionMask[i][: i + 1] = True

        self.attentions = [
            Attention(
                self.contextSize,
                self.embedDim,
                self.headCount,
                attentionMask,
            )
            for _ in range(layerCount)
        ]

        self.mlps = [Mlp(self.contextSize, self.embedDim) for _ in range(layerCount)]
        self.a = np.empty((contextSize, vocabSize))
        self.inputLength = 0
        self.loss = np.ones((self.contextSize, self.vocabSize))

    def save(self):
        attnqkv = []
        attnproj = []
        attng = []
        attnb = []
        mlpw0 = []
        mlpw1 = []
        mlpb0 = []
        mlpb1 = []
        mlpg = []
        mlpbeta = []

        for i in range(self.layerCount):
            attnqkv.append(self.attentions[i].qkv)
            attnproj.append(self.attentions[i].proj)
            attng.append(self.attentions[i].g)
            attnb.append(self.attentions[i].b)
            mlpw0.append(self.mlps[i].w[0])
            mlpw1.append(self.mlps[i].w[1])
            mlpb0.append(self.mlps[i].b[0])
            mlpb1.append(self.mlps[i].b[1])
            mlpg.append(self.mlps[i].g)
            mlpbeta.append(self.mlps[i].beta)

        with open("data/params.npz", "wb") as f:
            np.savez(
                f,
                attnqkv=np.hstack(attnqkv),
                attnproj=np.hstack(attnproj),
                attng=np.hstack(attng),
                attnb=np.hstack(attnb),
                mlpw0=np.hstack(mlpw0),
                mlpw1=np.hstack(mlpw1),
                mlpb0=np.hstack(mlpb0),
                mlpb1=np.hstack(mlpb1),
                mlpg=np.hstack(mlpg),
                mlpbeta=np.hstack(mlpbeta),
                b=self.b,
                g=self.g,
                pos=self.embedding.positions,
                words=self.embedding.words,
                allow_pickle=False,
            )

    def load(self):
        data = np.load("data/params.npz", allow_pickle=False)
        self.b = data["b"]
        self.g = data["g"]
        self.embedding.positions = data["pos"]
        self.embedding.words = data["words"]

        data = {
            k: np.split(v, self.layerCount, axis=-1)
            for k, v in data.items()
            if k not in ["b", "g", "pos", "words"]
        }
        for i in range(self.layerCount):
            self.attentions[i].qkv = data["attnqkv"][i]
            self.attentions[i].proj = data["attnproj"][i]
            self.attentions[i].g = data["attng"][i]
            self.attentions[i].b = data["attnb"][i]
            self.mlps[i].w = (data["mlpw0"][i], data["mlpw1"][i])
            self.mlps[i].b = (data["mlpb0"][i], data["mlpb1"][i])
            self.mlps[i].g = data["mlpg"][i]
            self.mlps[i].beta = data["mlpbeta"][i]
        # print(self.attentions[3].proj[23][35])

    def feedForward(self, input: str):
        self.tokens = np.array(encode(input, self.merges))[: self.contextSize]
        self.inputLength = len(self.tokens)
        self.tokens = np.pad(
            self.tokens, (0, max(0, self.contextSize - len(self.tokens)))
        )
        self.embedding.feedForward(self.tokens)
        lastLayer = layerNorm(self.embedding.a, self.g, self.b)
        # print(self.embedding.a)

        # attnTime = 0
        # mlpTime = 0
        for i in range(self.layerCount):
            # start = time.time()
            self.attentions[i].feedForward(lastLayer)
            lastLayer = self.attentions[i].a
            # attnTime += time.time() - start
            # start = time.time()
            self.mlps[i].feedForward((lastLayer))
            lastLayer = self.mlps[i].a
            # mlpTime += time.time() - start
        # print(attnTime, mlpTime)

        # print(lastLayer)
        self.embedding.decode(lastLayer)
        self.a = self.embedding.decoded
        # print(self.a)

    def backProp(self):
        probabilities = softmax(self.a)
        error = np.zeros((self.contextSize, self.vocabSize))

        # - 1/s(xi) * (s(xi) * (1 - s(xi) - sum(s(xj))))
        # = s(xi) + sum(s(xj)) - 1
        error = np.where(
            np.arange(self.contextSize).reshape(self.contextSize, 1) < self.inputLength,
            probabilities + probabilities.sum(-1).reshape(self.contextSize, 1) - 1,
            0,
        )
        self.embedding.decodeBackProp(error)
        # print(probabilities[1][self.tokens[1]])
        # print(error[1][self.tokens[1]])

    def getLoss(self):
        probabilities = softmax(self.a)
        self.loss = np.zeros((self.contextSize, self.vocabSize))

        for i in range(self.inputLength):
            self.loss[i][self.tokens[i]] = -np.log(probabilities[i][self.tokens[i]])

    def gradientDescent(self, learningRate: float, batchSize: int):
        self.embedding.gradientDescent(learningRate, batchSize)

    def getToken(self, index: int, T: float):
        probabilities = softmax(self.a[index], T=T)
        n = random.random()
        i = 0
        while n > 0:
            n -= probabilities[i]
            i += 1

        return i


if __name__ == "__main__":
    llm = LLM(50257, 768, 1024, 12, 12)
    start = time.time()
    try:
        llm.load()
    except Exception as e:
        print(e)
        print("failed to load previous params, creating new")
    else:
        print(f"loaded params {time.time() - start}s")
    # llm = LLM(50257, 8, 10, 2)

    # message = input("> ")
    # message = "hello world"
    # temperature = 1
    # i = 0
    # while True:
    #     if not i % 100:
    #         llm.save()
    #
    #     llm.feedForward(message)
    #     llm.backProp()
    #     new = decode([llm.getToken(llm.inputLength - 1, temperature)], llm.vocab)
    #     print(new, end="", flush=True)
    #     # message += new
    #     i += 1

    temperature = 1

    while True:
        llm.feedForward("hello world")
        new = decode([llm.getToken(llm.inputLength - 2, temperature)], llm.vocab)
        llm.backProp()
        llm.getLoss()
        llm.gradientDescent(1, 1)
        print("guess:", new, "loss", llm.loss.sum())
        llm.save()
