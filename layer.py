from __future__ import annotations
from typing import TYPE_CHECKING

import pygame

if TYPE_CHECKING:
    from network import Network
from node import Node


class Layer:
    def __init__(self, network: Network, layerNum: int, size: int, prevSize: int):
        self.network = network
        self.layerNum = layerNum
        self.size = size
        self.nodes: list[Node] = [Node(prevSize, self, network) for _ in range(size)]

    def draw(
        self, screen: pygame.Surface, font: pygame.font.Font, x: float, size: float
    ):
        gap = screen.get_height() / self.size

        for i in range(self.size):
            self.nodes[i].draw(screen, font, x, gap * (i + 0.5), min(size, gap * 0.325))
