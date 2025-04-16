from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from network import Network
from node import Node


class Layer:
    def __init__(self, network: Network, layerNum: int, size: int, prevSize: int):
        self.network = network
        self.layerNum = layerNum
        self.size = size
        self.nodes: list[Node] = [Node(prevSize, self, network) for _ in range(size)]
