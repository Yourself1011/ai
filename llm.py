from attention import Attention
from attentionHead import AttentionHead
from embedding import Embedding
from mlp import Mlp
from tokenizer import encode, load
import numpy as np


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

    def feedForward(self, input: str):
        tokens = np.array(encode(input, self.merges))[: self.contextSize]
        tokens = np.pad(tokens, (max(0, self.contextSize - len(tokens)), 0))
        self.embedding.feedForward(tokens)
        lastLayer = self.embedding.a

        for i in range(self.layerCount):
            self.attentions[i].feedForward(lastLayer)
            lastLayer = self.attentions[i].a
            self.mlps[i].feedForward((lastLayer))
            lastLayer = self.mlps[i].a

        print(lastLayer)


if __name__ == "__main__":
    llm = LLM(50257, 768, 1024, 12, 12)
    # llm = LLM(50257, 8, 10, 2)
    llm.feedForward("hello world")
