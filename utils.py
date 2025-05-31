import time
import numpy as np
import numpy.typing as npt

# smTime = 0


def sigmoid(x):
    # global smTime
    # start = time.time()
    result = 1 / (1 + np.exp(-x))
    # smTime += time.time() - start
    return result


def sigmoidPrime(x):
    # e^-x / (1 + e^-x)^2, or
    return sigmoid(x) * (1 - sigmoid(x))


geluCoefficient = np.sqrt(2 / np.pi)


def gelu(x):
    inside = geluCoefficient * (x + 0.044715 * x**3)
    tanh = np.tanh(inside)
    return 0.5 * x * (1 + tanh), tanh, inside


def layerNorm(x: npt.NDArray, g: npt.NDArray, b: npt.NDArray):
    # global smTime
    # start = time.time()
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    z = (x - mean) / np.sqrt(var + 1e-5)
    result = z * g + b
    # print(result.var(axis=-1))
    # print(result.mean(axis=-1))
    # smTime += time.time() - start
    return result, z, mean, var


def softmax(x, T: float = 1):
    # global smTime
    # start = time.time()
    adj = x / T if T != 1 else x
    exp = np.e ** (
        adj - adj.max(-1, keepdims=True)
    )  # we subtract the highest number, to keep values from getting too big
    res = exp / exp.sum(-1, keepdims=True)
    # smTime += time.time() - start
    return res
