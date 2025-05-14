from math import exp
import numpy as np


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoidPrime(x):
    # e^-x / (1 + e^-x)^2, or
    return sigmoid(x) * (1 - sigmoid(x))


geluCoefficient = np.sqrt(2 / np.pi)


def gelu(x):
    inside = geluCoefficient * (x + 0.044715 * x**3)
    tanh = np.tanh(inside)
    return 0.5 * x * (1 + tanh), tanh, inside


def layerNorm(x: np.typing.NDArray, g: np.typing.NDArray, b: np.typing.NDArray):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    z = (x - mean) / np.sqrt(var + 1e-5)
    result = z * g + b
    # print(result.var(axis=-1))
    # print(result.mean(axis=-1))
    return result, z, mean, var


def softmax(x, T: float = 1):
    adj = x / T
    exp = np.e ** (
        adj - adj.max(-1, keepdims=True)
    )  # we subtract the highest number, to keep values from getting too big
    return exp / exp.sum(-1, keepdims=True)
