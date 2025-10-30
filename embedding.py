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
        self.contextSize = contextSize
        self.error = np.zeros((contextSize, embedDim), dtype=np.float32)
        self.wordsError = np.zeros((vocabSize, embedDim), dtype=np.float32)
        self.decodeWordsError = np.zeros((vocabSize, embedDim), dtype=np.float32)
        self.positionsError = np.zeros((contextSize, embedDim), dtype=np.float32)
        self.a = np.empty((contextSize, embedDim), dtype=np.float32)
        self.decoded = np.empty(vocabSize, dtype=np.float32)
        self.inputLength = 0
        super().__init__()

    def feedForward(self, lastLayer: npt.NDArray):
        self.input = lastLayer
        # print(self.words[lastLayer].shape, self.positions.shape)
        self.a = self.words[lastLayer] + self.positions
        # print(self.words.max(0))

    def backProp(self, error: npt.NDArray):
        for j in range(self.input.shape[0] - 1):
            for i in range(self.contextSize):
                self.wordsError[self.input[j][i]] += error[j][i]
        for i in range(self.inputLength):
            self.wordsError[self.input[-1][i]] += error[-1][i]
        self.positionsError += error.sum(0)

    def decode(self, lastLayer: npt.NDArray):
        self.decodeInput = lastLayer
        self.decoded = lastLayer @ self.words.T

    def decodeBackProp(self, error: npt.NDArray):
        self.wordsError += (np.swapaxes(error, -1, -2) @ self.decodeInput).sum(axis=0)
        # print(error.shape, self.words.shape)
        self.error = error @ self.words

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

        # self.decodeWords = self.adamW(
        # "decodeWords", self.decodeWords, self.decodeWordsError, learningRate, t, mult
        # )

        # self.error = np.zeros((self.contextSize, self.embedDim))
        self.wordsError = np.zeros((self.vocabSize, self.embedDim))
        # self.decodeWordsError = np.zeros((self.vocabSize, self.embedDim))
        self.positionsError = np.zeros((self.contextSize, self.embedDim))
