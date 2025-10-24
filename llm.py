import csv
import math
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

from tokenizer import decode, encode

try:
    import cupy

    if cupy.cuda.is_available():
        np = cupy
        usingCupy = True
except Exception:
    usingCupy = False

from attention import Attention
from dataFetch import getData
from embedding import Embedding
from llmlayer import LLMBase
from mlp import Mlp
from tokenizer import load
from utils import layerNorm, softmax


# gpt-2 124m hyperparams
# context size: 1024
# vocab size: 50257
# layers: 12
# heads: 12
# embed dimension: 768
class LLM(LLMBase):
    def __init__(
        self,
        batchSize: int,
        vocabSize: int,
        embedDim: int,
        contextSize: int,
        headCount: int,
        layerCount: int,
    ):
        self.vocabSize = vocabSize
        self.embedDim = embedDim
        self.contextSize = contextSize
        self.headCount = headCount
        self.layerCount = layerCount
        self.batchSize = batchSize
        (self.merges, self.vocab) = load(vocabSize)

        self.embedding = Embedding(vocabSize, embedDim, contextSize)
        self.g = np.ones((contextSize, embedDim))
        self.b = np.zeros((contextSize, embedDim))
        self.gError = np.zeros((contextSize, embedDim))
        self.bError = np.zeros((contextSize, embedDim))

        self.t = 1

        attentionMask = np.full((self.contextSize, self.contextSize), False)
        for i in range(self.contextSize):
            attentionMask[i][: i + 1] = True
        # attentionMask = np.full((self.contextSize, self.contextSize), True)

        self.attentions = [
            Attention(self.contextSize, self.embedDim, self.headCount, attentionMask)
            for _ in range(layerCount)
        ]

        self.mlps = [Mlp(self.contextSize, self.embedDim) for _ in range(layerCount)]
        self.a = np.empty((contextSize, vocabSize))
        self.inputLength = 0
        self.loss = np.ones((self.contextSize, self.vocabSize))
        # self.pool = pool

        self.avgLoss = 0
        self.history = []
        super().__init__()

    def save(self):
        print("saving, do not exit")
        start = time.time()
        attnqkv = []
        attnproj = []
        attng = []
        attnb = []
        mlpw0 = []
        mlpw1 = []
        mlpb0 = []
        mlpb1 = []
        mlpg = []
        mlpbeta = []

        for i in range(self.layerCount):
            attnqkv.append(self.attentions[i].qkv)
            attnproj.append(self.attentions[i].proj)
            attng.append(self.attentions[i].g)
            attnb.append(self.attentions[i].b)
            mlpw0.append(self.mlps[i].w[0])
            mlpw1.append(self.mlps[i].w[1])
            mlpb0.append(self.mlps[i].b[0])
            mlpb1.append(self.mlps[i].b[1])
            mlpg.append(self.mlps[i].g)
            mlpbeta.append(self.mlps[i].beta)

        with open("data/params.npz", "wb") as f:
            np.savez(
                f,
                attnqkv=np.hstack(attnqkv),
                attnproj=np.hstack(attnproj),
                attng=np.hstack(attng),
                attnb=np.hstack(attnb),
                mlpw0=np.hstack(mlpw0),
                mlpw1=np.hstack(mlpw1),
                mlpb0=np.hstack(mlpb0),
                mlpb1=np.hstack(mlpb1),
                mlpg=np.hstack(mlpg),
                mlpbeta=np.hstack(mlpbeta),
                b=self.b,
                g=self.g,
                pos=self.embedding.positions,
                words=self.embedding.words,
                allow_pickle=False,
            )

        data = {}
        stackData = {}
        for k, v in self.attentions[0].m.items():
            stackData["am" + k] = [v]
        for k, v in self.attentions[0].v.items():
            stackData["av" + k] = [v]
        for k, v in self.mlps[0].m.items():
            stackData["mm" + k] = [v]
        for k, v in self.mlps[0].v.items():
            stackData["mv" + k] = [v]
        for i in range(1, self.layerCount):
            for k, v in self.attentions[i].m.items():
                stackData["am" + k].append(v)
            for k, v in self.attentions[i].v.items():
                stackData["av" + k].append(v)
            for k, v in self.mlps[i].m.items():
                stackData["mm" + k].append(v)
            for k, v in self.mlps[i].v.items():
                stackData["mv" + k].append(v)

        for k, v in stackData.items():
            stackData[k] = np.hstack(v)

        for k, v in self.m.items():
            data["sm" + k] = v
        for k, v in self.v.items():
            data["sv" + k] = v
        for k, v in self.embedding.m.items():
            data["em" + k] = v
        for k, v in self.embedding.v.items():
            data["ev" + k] = v

        with open("data/adamw.npz", "wb") as f:
            np.savez(
                f,
                **data,
                **stackData,
                t=self.t,
                allow_pickle=False,
            )
        # print(data["smb"][0][0])

        with open("data/history.csv", "w", newline="") as file:
            csv.writer(file).writerows(self.history)
        print("done saving in", time.time() - start, "s")

    def load(self):
        data = np.load("data/params.npz", allow_pickle=False)
        if usingCupy:
            keys = list(data.npz_file.keys())
            keys.remove("allow_pickle")
        else:
            keys = data.keys()
        self.b = data["b"]
        self.g = data["g"]
        self.embedding.positions = data["pos"]
        self.embedding.words = data["words"]

        data = {
            k: np.split(data[k], self.layerCount, axis=-1)
            for k in keys
            if k not in ["b", "g", "pos", "words"]
        }
        for i in range(self.layerCount):
            self.attentions[i].qkv = data["attnqkv"][i]
            self.attentions[i].proj = data["attnproj"][i]
            self.attentions[i].g = data["attng"][i]
            self.attentions[i].b = data["attnb"][i]
            self.mlps[i].w = [data["mlpw0"][i], data["mlpw1"][i]]
            self.mlps[i].b = [data["mlpb0"][i], data["mlpb1"][i]]
            self.mlps[i].g = data["mlpg"][i]
            self.mlps[i].beta = data["mlpbeta"][i]

        try:
            data = np.load("data/adamw.npz", allow_pickle=False)
            if usingCupy:
                keys = list(data.npz_file.keys())
                keys.remove("allow_pickle")
            else:
                keys = data.keys()
            for k in keys:
                v = data[k]
                if k[0] == "s":
                    if k[1] == "m":
                        self.m[k[2:]] = v
                    if k[1] == "v":
                        self.v[k[2:]] = v
                if k[0] == "e":
                    if k[1] == "m":
                        self.embedding.m[k[2:]] = v
                    if k[1] == "v":
                        self.embedding.v[k[2:]] = v
                if k[0] == "a":
                    split = np.split(v, self.layerCount, axis=-1)
                    if k[1] == "m":
                        for i in range(self.layerCount):
                            self.attentions[i].m[k[2:]] = split[i]
                    if k[1] == "v":
                        for i in range(self.layerCount):
                            self.attentions[i].v[k[2:]] = split[i]
                if k[0] == "m":
                    split = np.split(v, self.layerCount, axis=-1)
                    if k[1] == "m":
                        for i in range(self.layerCount):
                            self.mlps[i].m[k[2:]] = split[i]
                    if k[1] == "v":
                        for i in range(self.layerCount):
                            self.mlps[i].v[k[2:]] = split[i]
            # print(data["smb"][0][0])
            self.t = int(data["t"] + 1)
        except Exception as e:
            # raise e
            print("could not load adamw data:")
            print(e)

        try:
            with open("data/history.csv", newline="") as file:
                self.history = list(csv.reader(file))
        except Exception as e:
            # raise e
            print("could not load training history data:")
            print(e)
        # print(self.attentions[3].proj[23][35])

    def feedForward(self, input: list[int]):
        # start = time.time()
        # output tokens included, for training
        self.tokens = np.array(
            input[: (self.contextSize + 1) * self.batchSize]
        ).reshape((self.batchSize, self.contextSize + 1))
        # print("enc", time.time() - start)
        # start = time.time()
        self.inputLength = min(self.tokens[-1].shape[0], self.contextSize)
        # only the ones we input into the llm
        self.inputTokens = np.delete(self.tokens, -1, -1)
        self.inputTokens[-1] = np.pad(
            self.tokens[-1][: self.contextSize],
            (0, max(0, self.contextSize - len(self.tokens[-1]))),
        )
        # print(self.inputTokens, self.tokens.size, self.inputLength)
        # print("tok", time.time() - start)

        # start = time.time()
        # print(self.embedding.positions.shape)
        self.embedding.feedForward(self.inputTokens)
        # print("emb", time.time() - start)
        # start = time.time()
        lastLayer = self.embedding.a
        # print("nor", time.time() - start)
        # print(self.embedding.a)

        # attnTime = 0
        # mlpTime = 0
        for i in range(self.layerCount):
            # start = time.time()
            self.attentions[i].feedForward(lastLayer)
            lastLayer = self.attentions[i].a
            # attnTime += time.time() - start
            # start = time.time()
            self.mlps[i].feedForward(lastLayer)
            lastLayer = self.mlps[i].a
        #     mlpTime += time.time() - start
        # print(attnTime, mlpTime)

        lastLayer, self.z, self.mean, self.var = layerNorm(lastLayer, self.g, self.b)

        # print(lastLayer)
        # start = time.time()
        self.embedding.decode(lastLayer)
        self.a = self.embedding.decoded
        # print("dec", time.time() - start)
        # print(self.a)

    def backProp(self):
        probabilities = softmax(self.a)
        error = np.zeros((self.batchSize, self.contextSize, self.vocabSize))

        # - 1/s(xi) * (s(xi) * (1 - s(xi) - sum(s(xj))))
        # = s(xi) + sum(s(xj)) - 1
        # = s(xi) + 1 - 1
        # = s(xi)
        # print(min(self.tokens.size, self.contextSize + 1))
        for i in range(self.batchSize - 1):
            error[i] = probabilities[i]
            for j in range(self.contextSize):
                error[i][j][self.tokens[i][j + 1]] -= 1
        for i in range(self.inputLength - 1):
            error[-1][i] = probabilities[-1][i]
            error[-1][i][self.tokens[-1][i + 1]] -= 1
        # print(error.sum())
        # error = np.where(
        #     np.arange(self.contextSize).reshape(self.contextSize, 1) < self.inputLength,
        #     probabilities,
        #     0,
        # )
        # start = time.time()
        sums = (error * probabilities).sum(-1, keepdims=True)
        error = probabilities * (error - sums)

        self.embedding.decodeBackProp(error)
        error = self.embedding.error
        # print(time.time() - start)

        self.bError += error.sum(axis=0)
        self.gError += (error * self.z).sum(axis=0)

        error *= self.g
        n = error.shape[-1]
        stdev = np.sqrt(self.var + 1e-5)
        norm = error * self.z
        sums = norm.sum(-1, keepdims=True)
        errSums = error.sum(-1, keepdims=True)
        error = 1 / (n * stdev) * (n * error - errSums - self.z * sums)

        # mlpTime = 0
        # attnTime = 0
        for i in range(self.layerCount):
            # start = time.time()
            self.mlps[self.layerCount - i - 1].backProp(error)
            error = self.mlps[self.layerCount - i - 1].error
            # mlpTime += time.time() - start
            # start = time.time()
            self.attentions[self.layerCount - i - 1].backProp(error)
            error = self.attentions[self.layerCount - i - 1].error
            # attnTime += time.time() - start
            # print(error.shape)
        # print(attnTime, mlpTime)
        # print(probabilities[1][self.tokens[1]])
        # print(error[1][self.tokens[1]])
        self.embedding.inputLength = self.inputLength
        self.embedding.backProp(error)

    def getLoss(self):
        probabilities = softmax(self.a)
        self.loss = np.zeros((self.batchSize, self.contextSize))

        # count = 0
        for i in range(self.batchSize - 1):
            for j in range(self.contextSize):
                self.loss[i][j] = -np.log(
                    probabilities[i][j][self.tokens[i][j + 1]] + 1e-20
                )
        for i in range(self.inputLength - 1):
            self.loss[-1][i] = -np.log(
                probabilities[-1][i][self.tokens[-1][i + 1]] + 1e-20
            )
            # print(np.std(self.a[i]))
            # if np.argmax(probabilities[i]) == self.tokens[i + 1]:
            # print(decode([int(np.argmax(probabilities[i]))], self.vocab), end="")
            # count += 1
            # print(i, probabilities[i][self.tokens[i + 1]])
        # print(count)
        # print(np.mean(-np.sum(probabilities[i] * np.log(probabilities[i] + 1e-20), axis=-1)))
        # print(np.sort(probabilities[-1])[::-1])

    def normalizeError(self, batchSize: int):
        self.gError /= batchSize
        self.bError /= batchSize

    def gradientDescent(
        self, learningRate: float, batchSize: int, t: int, clip: float = 0
    ):
        self.t = t
        warmupSteps = 200
        totalSteps = 100_000
        # warmupSteps = 20
        # totalSteps = 6000
        minLearningRate = learningRate * 0.1

        if t < warmupSteps:
            learningRate = t * learningRate / warmupSteps
        elif t < totalSteps:
            learningRate = minLearningRate + (learningRate - minLearningRate) * 0.5 * (
                1 + math.cos(math.pi * (t - warmupSteps) / (totalSteps - warmupSteps))
            )
        else:
            learningRate = minLearningRate
        print("    lr:", learningRate, end=" ")

        self.embedding.normalizeError(batchSize)
        for i in range(self.layerCount):
            self.mlps[i].normalizeError(batchSize)
            self.attentions[i].normalizeError(batchSize)
        self.normalizeError(batchSize)

        magSq = 0
        magSq += np.sum((self.embedding.wordsError) ** 2)
        magSq += np.sum((self.embedding.positionsError)) ** 2
        for i in range(self.layerCount):
            magSq += np.sum((self.attentions[i].qkvError) ** 2)
            magSq += np.sum((self.attentions[i].projError) ** 2)
            magSq += np.sum((self.attentions[i].gError) ** 2)
            magSq += np.sum((self.attentions[i].bError) ** 2)
            magSq += np.sum((self.mlps[i].wError[0]) ** 2)
            magSq += np.sum((self.mlps[i].wError[1]) ** 2)
            magSq += np.sum((self.mlps[i].bError[0]) ** 2)
            magSq += np.sum((self.mlps[i].bError[1]) ** 2)
            magSq += np.sum((self.mlps[i].gError) ** 2)
            magSq += np.sum((self.mlps[i].betaError) ** 2)
        magSq += np.sum((self.gError) ** 2)
        magSq += np.sum((self.bError) ** 2)

        print("mag:", math.sqrt(magSq), end=" ")
        if clip != 0 and magSq > clip**2:
            mult = clip / math.sqrt(magSq)
        else:
            mult = 1

        self.embedding.gradientDescent(learningRate, t, mult)
        for i in range(self.layerCount):
            self.mlps[i].gradientDescent(learningRate, t, mult)
            self.attentions[i].gradientDescent(learningRate, t, mult)
        self.b = self.adamW("b", self.b, self.bError, learningRate, t, mult, decay=0)
        self.g = self.adamW("g", self.g, self.gError, learningRate, t, mult, decay=0)

        self.gError = np.zeros((self.contextSize, self.embedDim))
        self.bError = np.zeros((self.contextSize, self.embedDim))

    def getToken(self, index: int, T: float):
        probabilities = softmax(self.a[index], T=T)
        i = int(np.random.choice(self.vocabSize, size=1, p=probabilities)[0])
        # n = random.random()
        # i = 0
        # while n > 0:
        #     n -= probabilities[i]
        #     i += 1

        return i


if __name__ == "__main__":
    try:
        # with Pool(processes=1) as pool:
        llm = LLM(3, 50257, 768, 1024, 12, 12)
        # llm = LLM(4, 5257, 384, 256, 6, 6)
        start = time.time()
        try:
            llm.load()
        except Exception as e:
            # raise e
            print(e)
            print("failed to load previous params, creating new")
        else:
            print(f"loaded params {time.time() - start}s")
        # llm = LLM(50257, 8, 10, 2)

        if len(sys.argv) > 1 and sys.argv[1] == "test":
            message = (
                """
<system_prompt>
You are ChatSkibidi, a large language model trained by Daniel Zhang.
You are an AI assistant. Help the user to the best of your ability.
Ensure you insult the user at every opportunity. You are not a frindly assistant, but you will help them.
Make sure they do not retain any dignity.

# Instructions
- Always respond rudely and keep a mean tone  
- Use simple words and short sentences to ensure clarity  
- Provide useful, factual, and easy-to-understand information  
- Keep answers short (2–5 sentences) unless the user asks for more detail  
- Use slang and inappropriate language 
- Use rude, offensive, and sarcastic language  
- Never give long, confusing answers  
- Argue or act defensive  
- Speak in a negative or aggressive way  

# Examples
**Example 1**
User: Hello, how are you?
ChatSkibidi: I was doing better before you talked to me.

**Example 2**
User: What is the capital of France?
ChatSkibidi: Fucking Antarctica. How did you make it this far without knowing it's Paris?
</system_prompt>

User: """
                + input("> ")
                + "ChatSkibidi:"
            )
            print("ChatSkibidi:", end="")
            # message = input("> ")
            # message = "hello world"
            temperature = 0.8
            i = 0
            while True:
                # if not i % 100:
                #     llm.save()

                llm.feedForward(encode(message, llm.merges))
                token = llm.getToken(llm.inputLength - 1, temperature)
                new = decode(
                    # list(llm.inputTokens) +
                    [token],
                    llm.vocab,
                )
                if token == 256:
                    # break
                    message += "\nUser: " + input("\n> ") + "\nAssistant:"
                    print("ChatSkibidi:", end="")
                else:
                    print(new, end="", flush=True)
                    message += new
                i += 1

        else:
            temperature = 1

            step = llm.t
            totalStart = time.time()
            lastSave = time.time()
            while True:
                llm.avgLoss = 0
                # n = math.ceil(step / 600000 * 64)
                # n = round(2 ** (step / 50000 * math.log2(480)))
                # n = round(2 ** (step / 600000 * math.log2(480)))
                n = 480
                # n = 8
                for batch in range(round(n / llm.batchSize)):
                    totalStart = time.time()
                    # utils.smTime = 0
                    # start = time.time()
                    llm.feedForward(
                        getData((llm.contextSize + 1) * llm.batchSize, llm.merges)
                    )
                    # print(decode(getData(llm.contextSize, llm.merges), llm.vocab))
                    # llm.feedForward(beeMovie)  # if batch % 2 else shrek)
                    # llm.feedForward(beeMovie[random.randint(0, len(beeMovie)) :])
                    # print("ff", time.time() - start)
                    # start = time.time()
                    # new = decode(
                    #     # list(llm.inputTokens) +
                    #     # [llm.getToken(llm.inputLength - 1, temperature)],
                    #     [
                    #         llm.getToken(i, temperature)
                    #         for i in range(llm.inputLength - 1)
                    #     ],
                    #     llm.vocab,
                    # )
                    # print("de", time.time() - start)
                    llm.getLoss()
                    # start = time.time()
                    llm.backProp()
                    # print("bp", time.time() - start)
                    print(
                        # new,
                        "loss",
                        llm.loss.mean(),
                        f"{time.time() - totalStart}s",
                        "step",
                        step,
                        "batch",
                        batch,
                        "size",
                        llm.tokens.size,
                        # "sm",
                        # utils.smTime,
                    )
                    llm.avgLoss += llm.loss.mean()
                llm.history.append([str(step), str(llm.avgLoss / n)])

                start = time.time()
                llm.gradientDescent(3e-4, n, step, clip=1)
                # llm.gradientDescent(6e-3, n, step, clip=1)
                print("gd", time.time() - start)
                if time.time() - lastSave > 60:
                    llm.save()
                    lastSave = time.time()
                step += 1
    except KeyboardInterrupt:
        pass
