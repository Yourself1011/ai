from network import Network
from random import randint
import pygame

iterations = 10000
batchSize = 32
tests = 32
# iterations = 2
# batchSize = 1
# tests = 32
learningRate = 0.1

network = Network([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5])
# network = Network([2, 2, 1])

for i in range(iterations):
    avgError = 0
    for j in range(batchSize):
        a = randint(0, 0b1111)
        b = randint(0, 0b1111)
        # a = 0b0001
        # b = 0b0001
        result = a + b
        network.feedForward(
            input=[float(x) for x in f"{a:04b}"] + [float(x) for x in f"{b:04b}"]
        )
        network.backPropagate([float(x) for x in f"{result:05b}"])

        # a = randint(0, 1)
        # b = randint(0, 1)
        # result = a ^ b
        # network.feedForward([a, b])
        # network.backPropagate([result])

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
    a = randint(0, 0b1111)
    b = randint(0, 0b1111)
    # a = 0b0001
    # b = 0b0001
    result = a + b
    network.feedForward(
        input=[float(x) for x in f"{a:04b}"] + [float(x) for x in f"{b:04b}"]
    )

    # a = randint(0, 1)
    # b = randint(0, 1)
    # result = a ^ b
    # network.feedForward([a, b])

    # for layer in network.layers:
    #     print([x.a for x in layer.nodes])

    outputNodes = [x.a for x in network.layers[-1].nodes]
    output = 0
    for bit in outputNodes:
        output = (output << 1) | round(bit)

    print(a, b, result, output, outputNodes)
    # print(input, output, outputNodes)

pygame.init()
screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
font = pygame.font.SysFont("Poppins", 18)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL:
            for node in network.layers[0].nodes:
                if (
                    pygame.Vector2(pygame.mouse.get_pos()).distance_squared_to(
                        pygame.Vector2(node.x, node.y)
                    )
                    < node.size**2
                ):
                    node.a += event.y / 100
                    node.a = max(min(node.a, 1), 0)
                    network.feedForward()

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    network.draw(screen, font)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()
