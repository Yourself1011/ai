from embedding import Embedding
from tokenizer import encode, load
import numpy as np


class LLM:
    def __init__(self):
        (self.merges, self.vocab) = load()
        self.embedding = Embedding(50257, 768)

    def feedForward(self, input: str):
        tokens = np.array(encode(input, self.merges))[:1024]
        tokens = np.pad(tokens, (max(0, 1024 - len(tokens)), 0))
        self.embedding.feedForward(tokens)
        print(self.embedding.a)


if __name__ == "__main__":
    llm = LLM()
    llm.feedForward("hello world")
