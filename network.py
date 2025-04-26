from __future__ import annotations
import random
from typing import TYPE_CHECKING

from utils import sigmoid

if TYPE_CHECKING:
    import pygame
from layer import Layer
import numpy as np


class Network:
    def __init__(self, sizes: list[int]):
        self.layers = [
            Layer(self, i, sizes[i], sizes[i - 1] if i > 0 else 0)
            for i in range(len(sizes))
        ]
        self.maxSize = max(sizes)
        self.sizes = sizes
        self.weights = np.zeros((len(sizes), self.maxSize, self.maxSize))
        for i in range(1, len(sizes)):
            for j in range(sizes[i]):
                for k in range(sizes[i - 1]):
                    self.weights[i, j, k] = random.gauss()
        # print(self.weights)

        self.biases = np.zeros((len(sizes), self.maxSize))
        self.errors = np.zeros((len(sizes), self.maxSize))
        self.weightErrors = np.zeros((len(sizes), self.maxSize, self.maxSize))
        self.biasErrors = np.zeros((len(sizes), self.maxSize))
        self.z = np.zeros((len(sizes), self.maxSize))
        self.a = np.zeros((len(sizes), self.maxSize))

    def feedForward(self, input: list[float] = []):
        if len(input):
            input += [0] * (self.maxSize - len(input))
            self.a[0] = input

        for i in range(1, len(self.layers)):
            self.z[i] = (self.a[i - 1] * self.weights[i]).sum(axis=1) + self.biases[i]
            for j in range(self.sizes[i]):
                self.a[i, j] = sigmoid(self.z[i, j])
        # print(self.a)
        # print(self.z)
        # print(self.weights)
        # print(self.biases)

    def backPropagate(self, target: list[float]):
        target += [0] * (self.maxSize - len(target))
        # print(self.a)
        # print(self.weights)
        self.errors[-1] = 2 * (self.a[-1] - target)

        for i in range(len(self.layers) - 1, 0, -1):
            coefficient = self.errors[i] * self.a[i] * (1 - self.a[i])
            self.biasErrors[i] += coefficient
            coefficient = coefficient.reshape(self.maxSize, 1)
            self.errors[i - 1] += (coefficient * self.weights[i]).sum(axis=0)
            self.weightErrors[i] += coefficient * self.a[i - 1]
        # print("errors", errors)
        # print("errors", self.errors)
        # print("weight errors", weightErrors)
        # print("weight errors", self.weightErrors)
        # print("bias errors", biasErrors)
        # print("bias errors", self.biasErrors)

    def gradientDescent(self, learningRate: float, batchSize: int):
        self.weights -= self.weightErrors * learningRate / batchSize
        self.biases -= self.biasErrors * learningRate / batchSize
        self.errors = np.zeros((len(self.layers), self.maxSize))
        self.weightErrors = np.zeros((len(self.layers), self.maxSize, self.maxSize))
        self.biasErrors = np.zeros((len(self.layers), self.maxSize))

    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        gap = screen.get_width() / len(self.layers)

        for i in range(len(self.layers)):
            self.layers[i].draw(screen, font, gap * (i + 0.5), gap * 0.325)
