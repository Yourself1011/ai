import math
from pygame import KEYDOWN
from network import Network
from random import randint
from time import time

# iterations = 10000
batchSize = 32
tests = 32

iterations = 1
# batchSize = 1
# tests = 0

learningRate = 0.1

network = Network([8, 16, 5])
# network = Network([3, 3, 2])
# network = Network([2, 2, 1])

ffTime = 0
bpTime = 0

iterationsRun = 0


def train(iterations: int, training=True, verbose=False):
    global ffTime, bpTime
    passed = 0
    for _ in range(iterations):
        # full adder
        # a = randint(0, 1)
        # b = randint(0, 1)
        # c = randint(0, 1)
        # result = a + b + c
        # start = time()
        # network.feedForward(input=[a, b, c])
        # ffTime += time() - start
        # start = time()
        # if training:
        #     network.backPropagate([float(x) for x in f"{result:02b}"])
        # bpTime += time() - start

        # 4-bit adder
        a = randint(0, 0b1111)
        b = randint(0, 0b1111)
        # a = 0b0001
        # b = 0b0001
        result = a + b
        start = time()
        network.feedForward(
            input=[float(x) for x in f"{a:04b}"] + [float(x) for x in f"{b:04b}"]
        )
        ffTime += time() - start
        start = time()
        if training:
            network.backPropagate([float(x) for x in f"{result:05b}"])
        bpTime += time() - start

        # xor
        # a = randint(0, 1)
        # b = randint(0, 1)
        # result = a ^ b
        # network.feedForward([a, b])
        # if training:
        #     network.backPropagate([result])

        # for layer in network.layers:
        #     print([x.a for x in layer.nodes])

        outputNodes = network.a[-1][: network.sizes[-1]]
        output = 0
        for bit in outputNodes:
            output = (output << 1) | round(bit)

        if verbose:
            print(a, b, result, output, outputNodes)
            # print(input, output, outputNodes)
        if result == output:
            passed += 1

    if training:
        network.gradientDescent(learningRate, batchSize)

    return passed


if __name__ == "__main__":
    import pygame

    for i in range(iterations):
        train(batchSize)

        if not iterationsRun % 100:
            print(str(iterationsRun) + "/" + str(iterations))
        # print()
        iterationsRun += 1

    passed = train(tests, training=False, verbose=True)

    if tests:
        print(f"{passed}/{tests} tests passed ({round(passed / tests * 100, 2)}%)")
    print(ffTime, bpTime)

    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    running = True
    font = pygame.font.SysFont("Poppins", 18)
    training = False
    speed = 128

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
                        network.a[node.layerNum][node.nodeNum] += event.y / 100
                        network.a[node.layerNum][node.nodeNum] = max(
                            min(network.a[node.layerNum][node.nodeNum], 1), 0
                        )
                        network.feedForward()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    training = not training
                elif event.key == pygame.K_n:
                    train(batchSize)

                    if not iterationsRun % 100:
                        print(str(iterationsRun) + "/" + str(iterations))
                    iterationsRun += 1
                elif event.key == pygame.K_EQUALS:
                    speed *= 2
                elif event.key == pygame.K_MINUS:
                    speed //= 2
                    speed = max(speed, 1)

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")

        text = font.render(f"{iterationsRun} {speed}", True, [255, 255, 255])
        screen.blit(text, (5, 1))
        network.draw(screen, font)

        if training:
            for _ in range(speed):
                train(batchSize)

                if not iterationsRun % 100:
                    print(str(iterationsRun) + "/" + str(iterations))
                if not iterationsRun % 1000:
                    passed = train(tests, training=False)

                    if tests:
                        print(
                            f"{passed}/{tests} tests passed ({round(passed / tests * 100, 2)}%)"
                        )
                iterationsRun += 1

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

    pygame.quit()
