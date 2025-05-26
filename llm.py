import random
import time
from typing import Dict
from attention import Attention
from attentionHead import AttentionHead
from embedding import Embedding
from llmlayer import LLMBase, Layer
from mlp import Mlp
from tokenizer import decode, encode, load
import numpy as np
from utils import layerNorm, softmax
from multiprocessing import Pool


# gpt-2 124m hyperparams
# context size: 1024
# vocab size: 50257
# layers: 12
# heads: 12
# embed dimension: 768
class LLM(LLMBase):
    def __init__(
        self,
        vocabSize: int,
        embedDim: int,
        contextSize: int,
        headCount: int,
        layerCount: int,
        pool,
    ):
        self.vocabSize = vocabSize
        self.embedDim = embedDim
        self.contextSize = contextSize
        self.headCount = headCount
        self.layerCount = layerCount
        (self.merges, self.vocab) = load(vocabSize)

        self.embedding = Embedding(vocabSize, embedDim, contextSize)
        self.g = np.ones((contextSize, embedDim))
        self.b = np.zeros((contextSize, embedDim))
        self.gError = np.ones((contextSize, embedDim))
        self.bError = np.zeros((contextSize, embedDim))

        self.t = 1

        attentionMask = np.full((self.contextSize, self.contextSize), False)
        for i in range(self.contextSize):
            attentionMask[i][: i + 1] = True

        self.attentions = [
            Attention(
                self.contextSize, self.embedDim, self.headCount, attentionMask, pool
            )
            for _ in range(layerCount)
        ]

        self.mlps = [Mlp(self.contextSize, self.embedDim) for _ in range(layerCount)]
        self.a = np.empty((contextSize, vocabSize))
        self.inputLength = 0
        self.loss = np.ones((self.contextSize, self.vocabSize))
        self.pool = pool
        super().__init__()

    def save(self):
        print("saving, do not exit")
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
                t=self.t,
                allow_pickle=False,
            )
        print("done saving")

    def load(self):
        data = np.load("data/params.npz", allow_pickle=False)
        self.b = data["b"]
        self.g = data["g"]
        self.embedding.positions = data["pos"]
        self.embedding.words = data["words"]
        if "t" in data:
            self.t = data["t"]
            self.t = 1  # change this after saving adamw stuff

        data = {
            k: np.split(v, self.layerCount, axis=-1)
            for k, v in data.items()
            if k not in ["b", "g", "pos", "words", "t"]
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
        # print(self.attentions[3].proj[23][35])

    def feedForward(self, input: list[int]):
        # start = time.time()
        # output tokens included, for training
        self.tokens = np.array(input[: self.contextSize + 1])
        # print("enc", time.time() - start)
        # start = time.time()
        self.inputLength = len(self.tokens)
        # only the ones we input into the llm
        self.inputTokens = np.pad(
            self.tokens[: self.contextSize],
            (0, max(0, self.contextSize - len(self.tokens))),
        )
        # print("tok", time.time() - start)

        # start = time.time()
        self.embedding.feedForward(self.inputTokens)
        # print("emb", time.time() - start)
        # start = time.time()
        lastLayer, self.z, self.mean, self.var = layerNorm(
            self.embedding.a, self.g, self.b
        )
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
            self.mlps[i].feedForward((lastLayer))
            lastLayer = self.mlps[i].a
            # mlpTime += time.time() - start
        # print(attnTime, mlpTime)

        # print(lastLayer)
        # start = time.time()
        self.embedding.decode(lastLayer)
        self.a = self.embedding.decoded
        # print("dec", time.time() - start)
        # print(self.a)

    def backProp(self):
        probabilities = softmax(self.a)
        error = np.zeros((self.contextSize, self.vocabSize))

        # - 1/s(xi) * (s(xi) * (1 - s(xi) - sum(s(xj))))
        # = s(xi) + sum(s(xj)) - 1
        # = s(xi) + 1 - 1
        # = s(xi)
        for i in range(self.inputLength - 1):
            error[i] = probabilities[i]
            error[i][self.tokens[i + 1]] -= 1
        # print(error.sum())
        # error = np.where(
        #     np.arange(self.contextSize).reshape(self.contextSize, 1) < self.inputLength,
        #     probabilities,
        #     0,
        # )
        self.embedding.decodeBackProp(error)
        error = self.embedding.error

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
        self.bError += error
        self.gError += error * self.z
        n = error.shape[-1]
        stdev = np.sqrt(self.var + 1e-5)
        error *= self.g * (1 / (n * stdev)) * (n - 1 - self.z**2)
        self.embedding.backProp(error)

    def getLoss(self):
        probabilities = softmax(self.a)
        self.loss = np.zeros((self.contextSize, self.vocabSize))

        for i in range(self.inputLength - 1):
            self.loss[i][self.tokens[i + 1]] = -np.log(
                probabilities[i][self.tokens[i + 1]] + 1e-20
            )
            # print(i, probabilities[i][self.tokens[i + 1]])

    def gradientDescent(self, learningRate: float, batchSize: int, t: int):
        self.embedding.gradientDescent(learningRate, batchSize, t)
        for i in range(self.layerCount):
            self.mlps[i].gradientDescent(learningRate, batchSize, t)
            self.attentions[i].gradientDescent(learningRate, batchSize, t)
        self.b -= self.adamW("b", self.b, self.bError, learningRate, t) / batchSize
        self.g -= self.adamW("g", self.g, self.gError, learningRate, t) / batchSize

        self.gError = np.ones((self.contextSize, self.embedDim))
        self.bError = np.zeros((self.contextSize, self.embedDim))

    def getToken(self, index: int, T: float):
        probabilities = softmax(self.a[index], T=T)
        n = random.random()
        i = 0
        while n > 0:
            n -= probabilities[i]
            i += 1

        return i - 1


if __name__ == "__main__":
    with Pool(processes=4) as pool:
        llm = LLM(50257, 768, 1024, 12, 12, pool)
        start = time.time()
        try:
            llm.load()
        except Exception as e:
            print(e)
            print("failed to load previous params, creating new")
        else:
            print(f"loaded params {time.time() - start}s")
        # llm = LLM(50257, 8, 10, 2)

        # message = input("> ")
        # # message = "hello world"
        # temperature = 1
        # i = 0
        # while True:
        #     # if not i % 100:
        #     #     llm.save()
        #
        #     llm.feedForward(encode(message, llm.merges))
        #     new = decode(
        #         # list(llm.inputTokens) +
        #         [llm.getToken(len(llm.inputTokens) - 1, temperature)],
        #         llm.vocab,
        #     )
        #     print(new, end="", flush=True)
        #     message += new
        #     i += 1

        temperature = 1
        beeMovieStr = """According to all known laws of aviation, there is no way a bee should be able to fly.
    Its wings are too small to get its fat little body off the ground.
    The bee, of course, flies anyway because bees don't care what humans think is impossible.
    Yellow, black. Yellow, black. Yellow, black. Yellow, black.
    Ooh, black and yellow!
    Let's shake it up a little.
    Barry! Breakfast is ready!
    Coming!
    Hang on a second.
    Hello?
    Barry?
    Adam?
    Can you believe this is happening?
    I can't.
    I'll pick you up.
    Looking sharp.
    Use the stairs, Your father paid good money for those.
    Sorry. I'm excited.
    Here's the graduate.
    We're very proud of you, son.
    A perfect report card, all B's.
    Very proud.
    Ma! I got a thing going here.
    You got lint on your fuzz.
    Ow! That's me!
    Wave to us! We'll be in row 118,000.
    Bye!
    Barry, I told you, stop flying in the house!
    Hey, Adam.
    Hey, Barry.
    Is that fuzz gel?
    A little. Special day, graduation.
    Never thought I'd make it.
    Three days grade school, three days high school.
    Those were awkward.
    Three days college. I'm glad I took a day and hitchhiked around The Hive.
    You did come back different.
    Hi, Barry. Artie, growing a mustache? Looks good.
    Hear about Frankie?
    Yeah.
    You going to the funeral?
    No, I'm not going.
    Everybody knows, sting someone, you die.
    Don't waste it on a squirrel.
    Such a hothead.
    I guess he could have just gotten out of the way.
    I love this incorporating an amusement park into our day.
    That's why we don't need vacations.
    Boy, quite a bit of pomp under the circumstances.
    Well, Adam, today we are men.
    We are!
    Bee-men.
    Amen!
    Hallelujah!
    Students, faculty, distinguished bees,
    please welcome Dean Buzzwell.
    Welcome, New Hive City graduating class of 9:15.
    That concludes our ceremonies And begins your career at Honex Industries!
    Will we pick our job today?
    I heard it's just orientation.
    Heads up! Here we go.
    Keep your hands and antennas inside the tram at all times.
    Wonder what it'll be like?
    A little scary.
    Welcome to Honex, a division of Honesco and a part of the Hexagon Group.
    This is it!
    Wow.
    Wow.
    We know that you, as a bee, have worked your whole life to get to the point where you can work for your whole life.
    Honey begins when our valiant Pollen Jocks bring the nectar to The Hive.
    Our top-secret formula is automatically color-corrected, scent-adjusted and bubble-contoured into this soothing sweet syrup with its distinctive golden glow you know as... Honey!
    That girl was hot.
    She's my cousin!
    She is?
    Yes, we're all cousins.
    Right. You're right.
    At Honex, we constantly strive to improve every aspect of bee existence.
    These bees are stress-testing a new helmet technology.
    What do you think he makes?
    Not enough.
    Here we have our latest advancement, the Krelman.
    What does that do?
    Catches that little strand of honey that hangs after you pour it.
    Saves us millions.
    Can anyone work on the Krelman?
    Of course. Most bee jobs are small ones.
    But bees know that every small job, if it's done well, means a lot.
    But choose carefully because you'll stay in the job you pick for the rest of your life.
    The same job the rest of your life? I didn't know that.
    What's the difference?
    You'll be happy to know that bees, as a species, haven't had one day off in 27 million years.
    So you'll just work us to death?
    We'll sure try.
    Wow! That blew my mind!
    "What's the difference?"
    How can you say that?
    One job forever?
    That's an insane choice to have to make.
    I'm relieved. Now we only have to make one decision in life.
    But, Adam, how could they never have told us that?
    Why would you question anything? We're bees.
    We're the most perfectly functioning society on Earth.
    You ever think maybe things work a little too well here?
    Like what? Give me one example.
    I don't know. But you know what I'm talking about.
    Please clear the gate. Royal Nectar Force on approach.
    Wait a second. Check it out.
    Hey, those are Pollen Jocks!
    Wow.
    I've never seen them this close.
    They know what it's like outside The Hive.
    Yeah, but some don't come back.
    Hey, Jocks!"""

        shrekStr = """
    Once upon a time there was a lovely princess. But she had an enchantment upon her of a fearful sort, which could only be broken by Love's first kiss. She was locked away in a castle guarded by a terrible fire breathing dragon. Many brave knights had attempted to free her from this dreadful prison, but none prevailed. She waited in the dragon's keep in the highest room of the tallest tower for her true love and true love's first kiss. Like that's ever going to happen. What a loony. Shrek Beware Stay out I think he's in here. All right. Lets get it! Hold on. Do you know what that thing can do to you? Yeah. He'll groan into your bones for his brains. Well actually that would be a giant. Now Ogres, huh, they are much worse. They'll make a soup from your freshly peeled skin. They'll chew your livers, squeeze the jelly from your eyes. Actually, it's quite good on toast. Back, back beast, back! I warned you! Right. This is the part, where you run away. Yeah! And stay out. Wanted. Fairytale creatures. Right, this one is full. Take it away. Give me that. Your fine days are over. -25 pieces of silver for the witch. Next. -Come on. Sit down there! And be quiet! This cage is so small. You wouldn't turn me in. I'll never be stubborn again. I can change. Please, give me another chance. Oh, shut up! Next. What do we got? This little wooden puppet. I'm not a puppet, I'm a real boy. Five shillings for the possessed toy. Take it away. No! Please, don't let them do it! Next. What do you got? Well, I've got a talking donkey! Right. Well that's good for ten schillings, if you can prove it. Oh, go ahead fella. Well? He's just a li..., just a little nervous. He's really quite a chatterbox. You boneheaded donkey! That's it. I have heard enough. Guards! No, no, he talks, he does! I can talk. I love to talk. I've talked to... Get her out of my sight! -No, no, I swear! Hey, I can fly. -He can fly! -He can fly! He can talk! -That's right, fool! Now I'm a flying, talking donkey! You might have seen house fly, maybe even a superfly. But I bet you ain't never seen a donkey fly! Seize him! Get him! This way! Hurry! You there. Ogre. -I. By the order of lord Farquaad. I am authorized to place you both under arrest. And transport you to designated resettlement facility. Oh really? You and what army? Can I say something to you? Listen, you were really, really something, back there. Incredible. Are you talking to... ...me? Yes, I was talking to you. Can I just tell you that you were really great back there with those guards. They thought that was all over there. And then you showed up and BAM. There was tripping on over themselves like babes in the woods. That really made me feel good to see that. Oh, that's great. Really. Man, it's good to be free. Now, why don't you go celebrate your freedom with your own friends? But I... I don't have any friends. And I'm not going out there by myself. Hey wait a minute. I have a great idea... I'll stick with you. You and me in green fighting machine. Together we'll scare the spin if anybody crosses us. Oh, a, that was really scary. Maybe you don't mine me saying. If that don't work, your breath will certainly do the job done, 'cause... you definitively need some tic-tac or something, 'cause your breath stinks! Man you've ??? my note! Just like the time... ...and then I ate some rotten berries. Man I had some strong gases leaking out of my but that day. Why are you following me? I'll tell you why. 'Cause I'm all alone, there is no one here, beside me. My problems have all gone. There's no one to derive me. But you got to have free ... -Stop singing! Well, it's no wonder, you don't have any friends. Wow! Only a true friend would be that truly honest. Listen! Little donkey. Take a look at me! What am I? A... ...really tall? No! I'm an Ogre. You know, grab your torch and pitchforks. Doesn't that bother you? Nope. Really? -Really really. Oh? Man, I like you. What's your name? A..., Shrek. Shrek?! But do you know, what I like about you, Shrek? You've got that kind of: "I don't care what nobody thinks of me" thing. I like that, I respect that, Shrek. You're all right. Uh, look at that. Who would wanna live in a place like that? That would be my home. Oh, it is lovely. Just beautiful. You know you're quite a decorator. It's amazing what you did with such a modest budget. I like that boulder. That is a nice boulder. I guess, you don't entertain much, do you? I like my privacy. You know I do to."""
        print("tokenizing")
        start = time.time()
        beeMovie = encode(beeMovieStr, llm.merges)
        shrek = encode(shrekStr, llm.merges)
        print(
            f"finished tokenizing in {time.time() - start}s {len(shrek)}, {len(beeMovie)} tokens"
        )

        i = llm.t
        totalStart = time.time()
        while True:
            # start = time.time()
            llm.feedForward(beeMovie if i % 2 else shrek)
            # llm.feedForward(beeMovie[random.randint(0, len(beeMovie)) :])
            # print("ff", time.time() - start)
            # start = time.time()
            new = decode(
                # list(llm.inputTokens) +
                [llm.getToken(len(llm.inputTokens) - 1, temperature)],
                llm.vocab,
            )
            # print("de", time.time() - start)
            llm.getLoss()
            # start = time.time()
            llm.backProp()
            # print("bp", time.time() - start)
            # start = time.time()
            llm.gradientDescent(1e-3, 1, i)
            llm.t = i
            # print("gd", time.time() - start)
            print(
                new,
                "loss",
                llm.loss.sum(),
                f"{time.time() - totalStart}s",
                "iteration",
                i,
            )
            totalStart = time.time()
            if not i % 5:
                llm.save()
            i += 1
