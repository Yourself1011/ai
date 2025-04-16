from layer import Layer


class Network:
    def __init__(self, sizes: list[int]):
        self.layers = [
            Layer(self, i, sizes[i], sizes[i - 1] if i > 0 else 0)
            for i in range(len(sizes))
        ]

    def feedForward(self, input: list[float]):
        for i in range(len(self.layers[0].nodes)):
            self.layers[0].nodes[i].a = input[i]

        for i in range(1, len(self.layers)):
            for node in self.layers[i].nodes:
                node.activate()
