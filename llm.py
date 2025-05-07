import random
import time
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

    def feedForward(self, input: str):
        tokens = np.array(encode(input, self.merges))[: self.contextSize]
        self.inputLength = len(tokens)
        tokens = np.pad(tokens, (0, max(0, self.contextSize - len(tokens))))
        self.embedding.feedForward(tokens)
        lastLayer = layerNorm(self.embedding.a, self.g, self.b)

        for i in range(self.layerCount):
            self.attentions[i].feedForward(lastLayer)
            lastLayer = self.attentions[i].a
            self.mlps[i].feedForward((lastLayer))
            lastLayer = self.mlps[i].a

        # print(lastLayer)
        self.embedding.decode(lastLayer)
        self.a = self.embedding.decoded
        # print(self.a.shape)

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
    # llm = LLM(50257, 8, 10, 2)
    # message = input()
    message = "hello world"
    while True:
        llm.feedForward(message)
        new = decode([llm.getToken(llm.inputLength - 1, 5)], llm.vocab)
        print(new, end="", flush=True)
        message += new
