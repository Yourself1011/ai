from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pygame
    from layer import Layer
    from network import Network

import math
from random import gauss
from utils import sigmoid, sigmoidPrime


class Node:
    def __init__(
        self,
        prevLayerSize: int,
        layer: Layer,
        layerNum: int,
        nodeNum: int,
        network: Network,
    ):
        # self.weights = [gauss() / 2 for _ in range(prevLayerSize)]
        # self.bias = 0
        # self.weightsError: list[float] = [0 for _ in range(prevLayerSize)]
        # self.biasError = 0
        # self.error: float = 0
        self.prevLayerSize = prevLayerSize
        self.layer = layer
        self.network = network
        self.z = 0
        self.a = 0
        self.layerNum = layerNum
        self.nodeNum = nodeNum

    # def backPropagate(self):
    #     # sigmoid prime
    #     # e^-x / (1 + e^-x)^2, or
    #     # sigmoid(x) * (1 - sigmoid(x)), or
    #     # a * (1 - a)
    #     coefficient = self.error * self.a * (1 - self.a)
    #
    #     for i in range(len(self.weights)):
    #         self.weightsError[i] += (
    #             coefficient * self.network.layers[self.layer.layerNum - 1].nodes[i].a
    #         )
    #
    #         self.network.layers[self.layer.layerNum - 1].nodes[i].error += (
    #             coefficient * self.weights[i]
    #         )
    #
    #     self.biasError += coefficient
    #
    # def update(self, learningRate: float, n: int):
    #     for i in range(len(self.weights)):
    #         self.weights[i] -= self.weightsError[i] / n * learningRate
    #
    #     self.bias -= self.biasError / n * learningRate
    #     self.weightsError: list[float] = [0 for _ in range(len(self.weightsError))]
    #     self.biasError = 0
    #     self.error = 0

    def draw(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        x: float,
        y: float,
        size: float,
    ):
        import pygame
        from pygame import gfxdraw

        self.x = x
        self.y = y
        self.size = size
        self.a = self.network.a[self.layerNum][self.nodeNum]
        self.weights = self.network.weights[self.layerNum][self.nodeNum]

        text = font.render(str(round(self.a, 3)), True, [255, 255, 255])
        screen.blit(text, (self.x - text.get_width() / 2, self.y + size + 1))
        # text = font.render(
        #     "b: " + str(round(self.network.biases[self.layerNum][self.nodeNum], 3)),
        #     True,
        #     [255, 255, 255],
        # )
        # screen.blit(text, (self.x - text.get_width() / 2, self.y + size + 19))

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

        if self.layer.layerNum != 0:
            for i in range(self.prevLayerSize):
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
                    width=abs(math.ceil(self.weights[i] * size / 50)),
                )
        # text = font.render(
        #     ",".join(
        #         [
        #             str(round(x, 3))
        #             for x in self.network.weights[self.layerNum][self.nodeNum]
        #         ]
        #     ),
        #     True,
        #     [255, 255, 255] if self.a < 0.5 else [0, 0, 0],
        # )
        # screen.blit(text, (self.x - text.get_width() / 2, self.y))
