import torch

x = torch.randn(5, 10, requires_grad=True)


def torchLayerNorm(x):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)

    z = (x - mean) / torch.sqrt(var + 1e-5)
    result = z
    # print(result.var(axis=-1))
    # print(result.mean(axis=-1))
    # smTime += time.time() - start
    return result, z, mean, var


y, z, mean, var = torchLayerNorm(x)
error = torch.randn_like(y)

pytorchGrad = torch.autograd.grad(y, x, error, retain_graph=True)[0]

n = error.shape[-1]
stdev = torch.sqrt(var + 1e-5).reshape((-1, 1))
# error *= self.g * (1 / (n * stdev)) * (n - 1 - self.z**2)
norm = error * z
sums = norm.sum(-1).reshape((-1, 1))
errSums = error.sum(-1).reshape((-1, 1))
error = 1 / (n * stdev) * (n * error - errSums - z * sums)

print(torch.abs(pytorchGrad - error))
