from llmlayer import Layer
import numpy as np


class Embedding(Layer):
    def __init__(self, vocabSize: int, embedDim: int, contextSize: int):
        self.embedDim = embedDim
        self.vocabSize = vocabSize
        self.words = np.random.normal(size=(vocabSize, embedDim))
        self.positions = np.random.normal(size=(contextSize, embedDim))
        self.contextSize = contextSize
        self.wordsError = np.zeros((vocabSize, embedDim))
        self.positionsError = np.zeros((contextSize, embedDim))
        self.a = np.empty((contextSize, embedDim))
        self.decoded = np.empty(vocabSize)

    def feedForward(self, lastLayer: np.typing.NDArray):
        self.input = lastLayer
        self.a = self.words[lastLayer] + self.positions
        # for i in range(self.contextSize):
        #     self.a[i] = self.words[lastLayer[i]] + self.positions[i]

    def backProp(self, error: np.typing.NDArray):
        for i in range(self.contextSize):
            self.wordsError[self.input[i]] += error[i]
            self.positionsError[i] += error[i]

    def decode(self, lastLayer: np.typing.NDArray):
        self.decodeInput = lastLayer
        self.decoded = lastLayer @ self.words.T

    def decodeBackProp(self, error: np.typing.NDArray):
        self.wordsError += (self.decodeInput.T @ error).T
        # self.error

    def gradientDescent(self, learningRate: float, batchSize: int):
        self.words -= self.wordsError * learningRate / batchSize
        self.positions -= self.positionsError * learningRate / batchSize
        self.wordsError = np.zeros((self.vocabSize, self.embedDim))
        self.positionsError = np.zeros((self.contextSize, self.embedDim))
