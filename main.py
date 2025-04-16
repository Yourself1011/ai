from network import Network
from random import randint

iterations = 10000
batchSize = 32
tests = 32
# iterations = 2
# batchSize = 32
learningRate = 0.1

# network = Network([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5])
network = Network([2, 2, 1])

for i in range(iterations):
    avgError = 0
    for j in range(batchSize):
        # a = randint(0, 0b1111)
        # b = randint(0, 0b1111)
        # result = a + b
        # network.feedForward(
        #     [float(x) for x in f"{a:04b}"] + [float(x) for x in f"{b:04b}"]
        # )
        # network.backPropagate([float(x) for x in f"{result:05b}"])

        a = randint(0, 1)
        b = randint(0, 1)
        result = a ^ b
        network.feedForward([a, b])
        network.backPropagate([result])

        # for layer in network.layers:
        #     print([x.a for x in layer.nodes])

        outputNodes = [x.a for x in network.layers[-1].nodes]
        output = 0
        for bit in outputNodes:
            output = (output << 1) | round(bit)

        # print(a, b, result, output, outputNodes)
        # print(input, output, outputNodes)
    # gradient descent
    for l in range(len(network.layers) - 1, 0, -1):
        for node in network.layers[l].nodes:
            node.update(learningRate, batchSize)

    if not i % 100:
        print(str(i) + "/" + str(iterations))
    # print()
for i in range(tests):
    # a = randint(0, 0b1111)
    # b = randint(0, 0b1111)
    # result = a + b
    # network.feedForward([float(x) for x in f"{a:04b}"] + [float(x) for x in f"{b:04b}"])

    a = randint(0, 1)
    b = randint(0, 1)
    result = a ^ b
    network.feedForward([a, b])

    # for layer in network.layers:
    #     print([x.a for x in layer.nodes])

    outputNodes = [x.a for x in network.layers[-1].nodes]
    output = 0
    for bit in outputNodes:
        output = (output << 1) | round(bit)

    print(a, b, result, output, outputNodes)
    # print(input, output, outputNodes)
