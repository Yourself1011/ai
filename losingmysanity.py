import math
import torch
import numpy as np
# sanity checks

contextSize = 10
embedDim = 20
lastLayer = torch.randn(contextSize, embedDim, requires_grad=True)
w = [
    torch.randn(embedDim, 4 * embedDim, requires_grad=True),
    torch.randn(embedDim * 4, embedDim, requires_grad=True),
]
b = [
    torch.randn(4 * embedDim, requires_grad=True),
    torch.randn(embedDim, requires_grad=True),
]
g = torch.ones((contextSize, embedDim))
beta = torch.zeros((contextSize, embedDim))


def sigmoid(x):
    # global smTime
    # start = time.time()
    result = 1 / (1 + torch.exp(-x))
    # smTime += time.time() - start
    return result


input = lastLayer
# start = time.time()
layer1 = lastLayer @ w[0] + b[0]
# gelu, tanh, inside = gelu(layer1)
# use sigmoid approximation instad of tanh
multiplied = layer1 * 1.702
sigmoidRes = sigmoid(multiplied)
gelu = layer1 * sigmoidRes
layer2 = gelu @ w[1] + b[1]


def torchLayerNorm(x):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True, unbiased=False)

    z = (x - mean) / torch.sqrt(var + 1e-5)
    result = z * g + beta
    # print(result.var(axis=-1))
    # print(result.mean(axis=-1))
    # smTime += time.time() - start
    return result, z, mean, var


# def npmlp(lastLayer, w, b, g, beta):
#     def sigmoid(x):
#         result = 1 / (1 + np.exp(-x))
#         # smTime += time.time() - start
#         return result
#
#     def layerNorm(x, g, b):
#         # global smTime
#         # start = time.time()
#         mean = x.mean(axis=-1, keepdims=True)
#         var = x.var(axis=-1, keepdims=True)
#
#         z = (x - mean) / np.sqrt(var + 1e-5)
#         result = z * g + b
#         # print(result.var(axis=-1))
#         # print(result.mean(axis=-1))
#         # smTime += time.time() - start
#         return result, z, mean, var
#
#     lastLayer = lastLayer.detach().numpy()
#     w[0] = w[0].detach().numpy()
#     b[0] = b[0].detach().numpy()
#     w[1] = w[1].detach().numpy()
#     b[1] = b[1].detach().numpy()
#     g = g.detach().numpy()
#     beta = beta.detach().numpy()
#     # start = time.time()
#     layer1 = lastLayer @ w[0] + b[0]
#     # gelu, tanh, inside = gelu(layer1)
#     # use sigmoid approximation instad of tanh
#     multiplied = layer1 * 1.702
#     sigmoid = sigmoid(multiplied)
#     gelu = layer1 * sigmoid
#     layer2 = gelu @ w[1] + b[1]
#     return layerNorm(layer2, g, beta)
#
#
# npRes, npZ, npMean, npVar = npmlp(lastLayer, w, b, g, beta)

y, z, mean, var = torchLayerNorm(layer2)
# print(npRes - y.detach().numpy())

error = torch.randn_like(y)

pytorchGrad = torch.autograd.grad(y, w[0], error, retain_graph=True)[0]


def npBackprop(input, error, g, z, multiplied, sigmoid, var, gelu, w, b):
    input = input.detach().numpy()
    error = error.detach().numpy()
    z = z.detach().numpy()
    g = g.detach().numpy()
    multiplied = multiplied.detach().numpy()
    sigmoid = sigmoid.detach().numpy()
    var = var.detach().numpy()
    gelu = gelu.detach().numpy()
    w0 = w[0].detach().numpy()
    w1 = w[1].detach().numpy()
    b0 = b[0].detach().numpy()
    b1 = b[1].detach().numpy()

    betaError = error
    gError = error * z

    # derivative of layer norm
    error *= g
    n = error.shape[-1]
    stdev = np.sqrt(var + 1e-5)
    norm = error * z
    sums = norm.sum(-1, keepdims=True)
    errSums = error.sum(-1, keepdims=True)
    error = 1 / (n * stdev) * (n * error - errSums - z * sums)

    # print(error.shape)
    bError1 = error.sum(0)
    # print((gelu.T @ error).sum())
    # print(gelu.shape, error.shape)
    wError1 = gelu.T @ error
    error = sigmoid * (1 + multiplied * (1 - sigmoid)) * (error @ w1.T)

    # print(error)
    bError0 = error.sum(0)
    # print(input.shape, error.shape)
    wError0 = input.T @ error
    # print(error.shape, w[0].shape)
    error = error @ w0.T
    # error += ( w[0] @ error.T ).T
    return error


npRes = npBackprop(input, error, g, z, multiplied, sigmoidRes, var, gelu, w, b)

betaError = error
gError = error * z

error *= g
n = error.shape[-1]
stdev = torch.sqrt(var + 1e-5)
norm = error * z
sums = norm.sum(-1, keepdim=True)
errSums = error.sum(-1, keepdim=True)
error = 1 / (n * stdev) * (n * error - errSums - z * sums)

bError1 = error.sum(0)
# print((gelu.T @ error).sum())
# print(gelu.shape, error.shape)
wError1 = gelu.T @ error
geluError = error @ w[1].T
error = sigmoidRes * (1 + multiplied * (1 - sigmoidRes)) * geluError
# print(error)
bError0 = error.sum(0)
# print(input.shape, error.shape)
wError0 = input.T @ error
# print(error.shape, w[0].shape)
error = error @ w[0].T

print(error.detach().numpy() - npRes)

print(torch.abs(pytorchGrad - wError0))
