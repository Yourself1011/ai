import torch as np

try:
    import cupy

    if cupy.cuda.is_available():
        np = cupy
except Exception:
    pass
import torch.types as npt

from llmlayer import Layer


class Embedding(Layer):
    def __init__(self, vocabSize: int, embedDim: int, contextSize: int):
        self.embedDim = embedDim
        self.vocabSize = vocabSize
        self.words = np.normal(0, 0.02, size=(vocabSize, embedDim), requires_grad=True)
        self.decodeWords = np.normal(
            0, 0.02, size=(vocabSize, embedDim), requires_grad=True
        )
        self.positions = np.normal(
            0, 0.02, size=(contextSize, embedDim), requires_grad=True
        )
        self.contextSize = contextSize
        self.error = np.zeros((contextSize, embedDim))
        self.wordsError = np.zeros((vocabSize, embedDim))
        self.decodeWordsError = np.zeros((vocabSize, embedDim))
        self.positionsError = np.zeros((contextSize, embedDim))
        self.a = np.empty((contextSize, embedDim), requires_grad=True)
        self.decoded = np.empty(vocabSize)
        super().__init__()

    def feedForward(self, lastLayer: npt.Tensor):
        self.input = lastLayer
        # print(self.words[lastLayer].shape, self.positions.shape)
        self.a = self.words[lastLayer] + self.positions
        # print(self.words.max(0))

    def backProp(self, error):
        # print(error.shape, self.error.shape)
        for i in range(self.contextSize):
            self.wordsError[self.input[i]] += error[i]
        self.positionsError += error

    def decode(self, lastLayer):
        self.decodeInput = lastLayer
        self.decoded = lastLayer @ self.words.T

    def decodeBackProp(self, error):
        self.wordsError += error.T @ self.decodeInput
        # self.decodeWordsError = error.T @ self.decodeInput
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

        self.decodeWords = self.adamW(
            "decodeWords",
            self.decodeWords,
            self.decodeWordsError,
            learningRate,
            t,
            mult,
        )

        # self.error = np.zeros((self.contextSize, self.embedDim))
        self.wordsError = np.zeros((self.vocabSize, self.embedDim))
        self.decodeWordsError = np.zeros((self.vocabSize, self.embedDim))
        self.positionsError = np.zeros((self.contextSize, self.embedDim))
