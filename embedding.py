from llmlayer import Layer
import numpy as np


class Embedding(Layer):
    def __init__(self, vocabSize: int, embedDim: int):
        self.embedDim = embedDim
        self.matrix = np.random.normal(size=(vocabSize, embedDim))
        self.a = np.array([])

    def feedForward(self, lastLayer: np.typing.NDArray):
        self.input = lastLayer
        self.a = np.zeros((len(lastLayer), self.embedDim))
        for i in range(len(lastLayer)):
            self.a[i] = self.matrix[lastLayer[i]]

    def backProp(self, error: np.typing.NDArray):
        self.error += error

    def gradientDescent(self, learningRate: float, batchSize: int):
        self.matrix -= self.error * learningRate / batchSize
