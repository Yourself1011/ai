import math
from numpy import require
import torch
# sanity checks

contextSize = 10
embedDim = 10
headCount = 2


attentionMask = torch.full((contextSize, contextSize), False)
for i in range(contextSize):
    attentionMask[i][: i + 1] = True


def softmax(x, T: float = 1):
    # global smTime
    # start = time.time()
    adj = x / T if T != 1 else x
    exp = torch.e ** (
        adj - adj.max(-1, keepdims=True).values
    )  # we subtract the highest number, to keep values from getting too big
    res = exp / exp.sum(-1, keepdims=True)
    # smTime += time.time() - start
    return res


lastLayer = torch.randn(contextSize, embedDim, requires_grad=True)
qkv = torch.randn(embedDim, embedDim * 3, requires_grad=True)
proj = torch.randn(embedDim, embedDim, requires_grad=True)

g = torch.randn(contextSize, embedDim, requires_grad=True)
b = torch.randn(contextSize, embedDim, requires_grad=True)

input = lastLayer
q, k, v = [
    torch.split(x, embedDim // headCount, dim=1)
    for x in torch.split(lastLayer @ qkv, embedDim, dim=1)
]


query = q[0]
key = k[0]
value = v[0]
attentionPattern = torch.where(
    attentionMask,
    # (
    #     torch.matmul(lastLayer, query)
    #     * torch.matmul(lastLayer, key).reshape(
    #         contextSize, 1, embedDim // headCount
    #     )
    # ).sum(2),
    query @ key.T,
    -torch.inf,
) / (math.sqrt(embedDim))

weights = softmax(attentionPattern)
# value = torch.matmul(lastLayer, torch.matmul(valueUp, valueDown))
# print("alskdjf")
# change = (
#     value.reshape(1, contextSize, embedDim)
#     * weights.reshape(contextSize, contextSize, 1)
# ).sum(1)

headA = weights @ value


query1 = q[1]
key1 = k[1]
value1 = v[1]
attentionPattern1 = torch.where(
    attentionMask,
    # (
    #     torch.matmul(lastLayer, query)
    #     * torch.matmul(lastLayer, key).reshape(
    #         contextSize, 1, embedDim // headCount
    #     )
    # ).sum(2),
    query1 @ key1.T,
    -torch.inf,
) / (math.sqrt(embedDim))

weights1 = softmax(attentionPattern1)
# value = torch.matmul(lastLayer, torch.matmul(valueUp, valueDown))
# print("alskdjf")
# change = (
#     value.reshape(1, contextSize, embedDim)
#     * weights.reshape(contextSize, contextSize, 1)
# ).sum(1)

headB = weights1 @ value1


def layerNorm(x, g, b):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True, unbiased=False)

    z = (x - mean) / torch.sqrt(var + 1e-5)
    result = z * g + b
    # print(result.var(axis=-1))
    # print(result.mean(axis=-1))
    # smTime += time.time() - start
    return result, z, mean, var


combined = torch.hstack([headA, headB])
preLN = combined @ proj
a, z, mean, var = layerNorm(preLN, g, b)

error = torch.randn_like(a)

pytorchGrad = torch.autograd.grad(a, lastLayer, error, retain_graph=True)[0]

bError = error
gError = error * z

# !! CHANGE !!
error *= g
# derivative of layer norm
n = error.shape[-1]
stdev = torch.sqrt(var + 1e-5).reshape((-1, 1))
norm = error * z
sums = norm.sum(-1).reshape((-1, 1))
errSums = error.sum(-1).reshape((-1, 1))
error = 1 / (n * stdev) * (n * error - errSums - z * sums)

projError = combined.T @ error

splitError = torch.split(error @ proj.T, embedDim // headCount, dim=1)
# print(splitError[0].shape)
qkvErrors = [[], [], []]

error = splitError[0]
valueError = weights.T @ error

error = error @ value.T
sums = (error * weights).sum(-1).reshape((-1, 1))
error = weights * (error - sums) / math.sqrt(embedDim)

queryError = error @ key
keyError = error.T @ query

qkvErrors[0].append(queryError)
qkvErrors[1].append(keyError)
qkvErrors[2].append(valueError)

error = splitError[1]
valueError = weights1.T @ error

error = error @ value1.T
sums = (error * weights1).sum(-1).reshape((-1, 1))
error = weights1 * (error - sums) / math.sqrt(embedDim)

queryError = error @ key1
keyError = error.T @ query1

qkvErrors[0].append(queryError)
qkvErrors[1].append(keyError)
qkvErrors[2].append(valueError)
# qkvErrors[0].append(
#     torch.zeros((contextSize, embedDim // headCount))
# )
# qkvErrors[1].append(
#     torch.zeros((contextSize, embedDim // headCount))
# )
# qkvErrors[2].append(
#     torch.zeros((contextSize, embedDim // headCount))
# )

error = torch.hstack([torch.hstack(x) for x in qkvErrors])
qkvError = input.T @ error
# print(error.shape, qkv.shape)
error = error @ qkv.T

print(torch.abs(pytorchGrad - error))
