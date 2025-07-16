import math
import torch
# sanity checks

contextSize = 10
embedDim = 10
query = torch.randn(contextSize, embedDim, requires_grad=True)
key = torch.randn(contextSize, embedDim, requires_grad=True)


attentionMask = torch.full((contextSize, contextSize), False)
for i in range(contextSize):
    attentionMask[i][: i + 1] = True

attentionPattern = torch.where(
    attentionMask,
    # (
    #     np.matmul(lastLayer, self.query)
    #     * np.matmul(lastLayer, self.key).reshape(
    #         contextSize, 1, self.embedDim // self.headCount
    #     )
    # ).sum(2),
    query @ key.T,
    -torch.inf,
) / (math.sqrt(embedDim))


def torchSoftmax(x, T: float = 1):
    # global smTime
    # start = time.time()
    adj = x / T if T != 1 else x
    exp = torch.e ** (
        adj - adj.max(-1, keepdims=True).values
    )  # we subtract the highest number, to keep values from getting too big
    res = exp / exp.sum(-1, keepdims=True)
    # smTime += time.time() - start
    return res


y = torchSoftmax(attentionPattern)
error = torch.randn_like(y)

pytorchGrad = torch.autograd.grad(y, key, error, retain_graph=True)[0]

sums = (error * y).sum(-1).reshape((-1, 1))
error = y * (error - sums) / math.sqrt(embedDim)
error = (query.T @ error).T
print(torch.abs(pytorchGrad - error))
