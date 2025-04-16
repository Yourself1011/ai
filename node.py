from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layer import Layer
    from network import Network

from random import gauss
from utils import sigmoid, sigmoidPrime


class Node:
    def __init__(self, prevLayerSize: int, layer: Layer, network: Network):
        self.weights = [gauss() for _ in range(prevLayerSize)]
        self.bias = 0
        self.weightsError: list[float] = [0 for _ in range(prevLayerSize)]
        self.biasError = 0
        self.error: float = 0
        self.layer = layer
        self.network = network
        self.z = 0
        self.a = 0

    def activate(self):
        self.z = (
            sum(
                self.network.layers[self.layer.layerNum - 1].nodes[i].a
                * self.weights[i]
                for i in range(len(self.weights))
            )
            + self.bias
        )

        # print(self.z, self.weights, self.bias)
        self.a = sigmoid(self.z)

    def backPropagate(self):
        coefficient = self.error * sigmoidPrime(self.z)

        for i in range(len(self.weights)):
            self.weightsError[i] += (
                coefficient * self.network.layers[self.layer.layerNum - 1].nodes[i].a
            )

            self.network.layers[self.layer.layerNum - 1].nodes[i].error += (
                coefficient * self.weights[i]
            )

        self.biasError += coefficient

    def update(self, learningRate: float, n: int):
        for i in range(len(self.weights)):
            self.weights[i] -= self.weightsError[i] / n * learningRate

        self.bias -= self.biasError / n * learningRate
        self.weightsError: list[float] = [0 for _ in range(len(self.weightsError))]
        self.biasError = 0
        self.error = 0
