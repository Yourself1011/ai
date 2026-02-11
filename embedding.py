import numpy as np

try:
    import cupy

    if cupy.cuda.is_available():
        np = cupy
except Exception:
    pass
import numpy.typing as npt

from llmlayer import Layer


class Embedding(Layer):
    def __init__(self, vocabSize: int, embedDim: int, contextSize: int):
        self.embedDim = embedDim
        self.vocabSize = vocabSize
        self.words = np.random.normal(
            0, 1 / np.sqrt(embedDim), size=(vocabSize, embedDim)
        ).astype(np.float32)
        self.decodeWords = np.random.normal(
            0, 1 / np.sqrt(embedDim), size=(vocabSize, embedDim)
        ).astype(np.float32)
        self.positions = np.random.normal(
            0, 1 / np.sqrt(embedDim), size=(contextSize, embedDim)
        ).astype(np.float32)
        self.words16 = self.words.astype(np.float16)
        self.positions16 = self.positions.astype(np.float16)

        self.contextSize = contextSize
        self.error = np.zeros((contextSize, embedDim), dtype=np.float16)
        self.wordsError = np.zeros((vocabSize, embedDim), dtype=np.float32)
        self.decodeWordsError = np.zeros((vocabSize, embedDim), dtype=np.float32)
        self.positionsError = np.zeros((contextSize, embedDim), dtype=np.float32)
        self.a = np.empty((contextSize, embedDim), dtype=np.float16)
        self.decoded = np.empty(vocabSize, dtype=np.float32)
        self.inputLength = 0
        super().__init__()

    def feedForward(self, lastLayer: npt.NDArray):
        self.input = lastLayer
        # print(self.words[lastLayer].shape, self.positions.shape)
        self.a = self.words16[lastLayer] + self.positions16
        # print(self.words.max(0))

    def backProp(self, error: npt.NDArray):
        for j in range(self.input.shape[0] - 1):
            self.wordsError[self.input[j]] += error.astype(np.float16)[j]
        self.wordsError[self.input[-1]] += error.astype(np.float16)[-1]
        self.positionsError += error.astype(np.float16).sum(0)

    def decode(self, lastLayer: npt.NDArray):
        self.decodeInput = lastLayer
        self.decoded = lastLayer @ self.words16.T

    def decodeBackProp(self, error: npt.NDArray):
        self.wordsError += (
            np.swapaxes(error, -1, -2) @ self.decodeInput.astype(np.float16)
        ).sum(axis=0)
        # print(error.shape, self.words.shape)
        self.error = error @ self.words16

    def normalizeError(self, batchSize: int):
        self.wordsError /= batchSize
        # self.decodeWordsError /= batchSize
        self.positionsError /= batchSize

    def gradientDescent(self, learningRate: float, t: int, mult: float):
        self.words = self.adamW(
            "words", self.words, self.wordsError, learningRate, t, mult
        )
        self.positions = self.adamW(
            "positions", self.positions, self.positionsError, learningRate, t, mult
        )

        self.words16 = self.words.astype(np.float16)
        self.positions16 = self.positions.astype(np.float16)

        # self.decodeWords = self.adamW(
        # "decodeWords", self.decodeWords, self.decodeWordsError, learningRate, t, mult
        # )

        # self.error = np.zeros((self.contextSize, self.embedDim))
        self.wordsError = np.zeros((self.vocabSize, self.embedDim))
        # self.decodeWordsError = np.zeros((self.vocabSize, self.embedDim))
        self.positionsError = np.zeros((self.contextSize, self.embedDim))
