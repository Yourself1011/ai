from random import randint
import time
from multiprocessing import Process, Queue

from tokenizer import encode
from getData import getBee, getMyData, wikiPage, pj

tokens = []
sampleSize = 100
buffer = []
process = Process()
queue = Queue()


def getData(amt: int, merges):
    global i, tokens, process, queue, buffer
    if len(tokens) == 0:
        tokens = [getTokens(merges) for _ in range(sampleSize)]

    return addToBuffer(merges, amt)


def getTokens(merges):
    start = time.time()
    filtered = pj()
    # print(filtered)
    new = encode(filtered, merges) + [256]
    print(
        "    got text in",
        time.time() - start,
        "s,",
        len(new),
        "tokens",
    )
    return new


def addToBuffer(merges, amt):
    global tokens
    buffer = []
    while len(buffer) < amt:
        idx = randint(0, sampleSize - 1)
        lenBuf = len(buffer)
        buffer += tokens[idx][: amt - lenBuf]
        del tokens[idx][: round((amt - lenBuf) * 3 / 4)]
        if len(tokens[idx]) == 0:
            tokens[idx] = getTokens(merges)

    return buffer


def asyncAddToBuffer(queue, merges, amt):
    queue.put(addToBuffer(merges, amt))
