from math import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoidPrime(x):
    # e^-x / (1 + e^-x)^2, or
    return sigmoid(x) * (1 - sigmoid(x))
