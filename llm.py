from attentionHead import AttentionHead
from embedding import Embedding
from tokenizer import encode, load
import numpy as np


# gpt-2 124m hyperparams
# context size: 1024
# vocab size: 50257
# layers: 12
# heads: 12
# embed dimension: 768
class LLM:
    def __init__(self, vocabSize: int, embedDim: int, contextSize: int, headCount: int):
        self.vocabSize = vocabSize
        self.embedDim = embedDim
        self.contextSize = contextSize
        self.headCount = headCount
        (self.merges, self.vocab) = load(vocabSize)
        self.embedding = Embedding(vocabSize, embedDim, contextSize)

    def feedForward(self, input: str):
        tokens = np.array(encode(input, self.merges))[: self.contextSize]
        tokens = np.pad(tokens, (max(0, self.contextSize - len(tokens)), 0))
        self.embedding.feedForward(tokens)

        attentionMask = np.full((self.contextSize, self.contextSize), False)
        for i in range(self.contextSize):
            attentionMask[i][: i + 1] = True
        attention = AttentionHead(
            self.contextSize, self.embedDim, self.headCount, attentionMask
        )
        print(self.embedding.a.shape)
        attention.feedForward(self.embedding.a)


if __name__ == "__main__":
    llm = LLM(50257, 768, 1024, 12)
    # llm = LLM(50257, 8, 10, 2)
    llm.feedForward("hello world")
