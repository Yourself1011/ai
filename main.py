from network import Network

network = Network([8, 8, 8, 5])

network.feedForward([0, 0, 0, 0, 0, 0, 0, 1])
print([x.a for x in network.layers[-1].nodes])
