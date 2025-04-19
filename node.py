from __future__ import annotations
from typing import TYPE_CHECKING

import pygame
from pygame import gfxdraw

if TYPE_CHECKING:
    from layer import Layer
    from network import Network

from random import gauss
from utils import sigmoid, sigmoidPrime


class Node:
    def __init__(self, prevLayerSize: int, layer: Layer, network: Network):
        self.weights = [gauss() / 2 for _ in range(prevLayerSize)]
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

    def draw(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        x: float,
        y: float,
        size: float,
    ):
        self.x = x
        self.y = y
        self.size = size

        text = font.render(str(round(self.a, 3)), True, [255, 255, 255])
        screen.blit(text, (self.x - text.get_width() / 2, self.y + size + 1))

        if self.layer.layerNum != 0:
            for i in range(len(self.weights)):
                target = self.network.layers[self.layer.layerNum - 1].nodes[i]
                pygame.draw.line(
                    screen,
                    [
                        min(
                            max(0, int((sigmoid(self.weights[i]) - 1) * -255)),
                            255,
                        ),
                        min(max(0, int((sigmoid(self.weights[i])) * 255)), 255),
                        0,
                    ],
                    pygame.Vector2(x, y),
                    pygame.Vector2(target.x, target.y),
                    # width=abs(round((sigmoid(self.weights[i]) - 0.5) * size / 5)),
                    width=abs(round(self.weights[i] * size / 100)),
                )

        gfxdraw.filled_circle(
            screen,
            round(x),
            round(y),
            round(size),
            [int(self.a * 255) for _ in range(3)],
        )
        gfxdraw.aacircle(
            screen, round(x), round(y), round(size), [255 for _ in range(3)]
        )
