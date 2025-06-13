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
        self.words = np.random.normal(size=(vocabSize, embedDim))
        self.positions = np.random.normal(size=(contextSize, embedDim))
        self.contextSize = contextSize
        self.error = np.zeros((contextSize, embedDim))
        self.wordsError = np.zeros((vocabSize, embedDim))
        self.positionsError = np.zeros((contextSize, embedDim))
        self.a = np.empty((contextSize, embedDim))
        self.decoded = np.empty(vocabSize)
        super().__init__()

    def feedForward(self, lastLayer: npt.NDArray):
        self.input = lastLayer
        self.a = self.words[lastLayer] + self.positions
        # for i in range(self.contextSize):
        #     self.a[i] = self.words[lastLayer[i]] + self.positions[i]

    def backProp(self, error: npt.NDArray):
        for i in range(self.contextSize):
            self.wordsError[self.input[i]] += error[i]
        self.positionsError += error

    def decode(self, lastLayer: npt.NDArray):
        self.decodeInput = lastLayer
        self.decoded = lastLayer @ self.words.T

    def decodeBackProp(self, error: npt.NDArray):
        # self.wordsError += (self.decodeInput.T @ error).T
        self.wordsError += error.T @ self.decodeInput
        # print(error.shape, self.decodeInput.shape)
        self.error += error @ self.words

    def normalizeError(self, batchSize: int):
        self.wordsError /= batchSize
        self.positionsError /= batchSize

    def gradientDescent(self, learningRate: float, t: int, mult: float):
        self.words -= self.adamW(
            "words", self.words, self.wordsError, learningRate, t, mult
        )
        self.positions -= self.adamW(
            "positions", self.positions, self.positionsError, learningRate, t, mult
        )

        self.error = np.zeros((self.contextSize, self.embedDim))
        self.wordsError = np.zeros((self.vocabSize, self.embedDim))
        self.positionsError = np.zeros((self.contextSize, self.embedDim))
