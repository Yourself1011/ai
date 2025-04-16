from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layer import Layer
    from network import Network

from random import gauss
from utils import sigmoid


class Node:
    def __init__(self, prevLayerSize: int, layer: Layer, network: Network):
        self.weights = [gauss() for _ in range(prevLayerSize)]
        self.bias = gauss()
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

        self.a = sigmoid(self.z)
