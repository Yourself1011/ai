import math
import time
import torch as np

try:
    import cupy

    if cupy.cuda.is_available():
        np = cupy
except Exception:
    pass
import torch.types as npt

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


geluCoefficient = math.sqrt(2 / math.pi)


def gelu(x):
    inside = geluCoefficient * (x + 0.044715 * x**3)
    tanh = np.tanh(inside)
    return 0.5 * x * (1 + tanh), tanh, inside


def layerNorm(x, g: npt.Tensor, b: npt.Tensor):
    # global smTime
    # start = time.time()
    mean = x.mean(dim=-1, keepdims=True)
    var = x.var(dim=-1, keepdims=True, unbiased=False)

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
        adj - adj.max(dim=-1, keepdims=True)[0]
    )  # we subtract the highest number, to keep values from getting too big
    res = exp / exp.sum(dim=-1, keepdims=True)
    # smTime += time.time() - start
    return res
