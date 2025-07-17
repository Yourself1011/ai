import math
import torch
# sanity checks

contextSize = 10
embedDim = 10
lastLayer = torch.randn(contextSize, embedDim, requires_grad=True)
w = (
    torch.randn(embedDim, 4 * embedDim, requires_grad=True),
    torch.randn(embedDim * 4, embedDim, requires_grad=True),
)
b = (
    torch.randn(4 * embedDim, requires_grad=True),
    torch.randn(embedDim, requires_grad=True),
)


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


error = torch.randn_like(layer2)

pytorchGrad = torch.autograd.grad(layer2, b[0], error, retain_graph=True)[0]

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
# error = error @ w[0].T
print(torch.abs(pytorchGrad - bError0))
