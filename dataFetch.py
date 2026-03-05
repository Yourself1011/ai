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
i = 0


def getData(amt: int, merges):
    global i, tokens, process, queue, buffer
    if len(tokens) == 0:
        tokens = [getTokens(merges) for _ in range(sampleSize)]

    if i + amt >= len(tokens):
        buffer = addToBuffer(merges, amt)
        chunk = tokens[i:] + buffer[: amt - (len(tokens) % amt)]
        tokens = buffer[amt - (len(tokens) % amt) :]

        i = 0
    else:
        chunk = tokens[i : i + amt]
        i += amt
    return chunk


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
        buffer += tokens[idx][: amt - len(buffer)]
        del tokens[idx][: round((amt - len(buffer)) * 3 / 4)]
        if len(tokens[idx]) == 0:
            tokens[idx] = getTokens(merges)

    return buffer


def asyncAddToBuffer(queue, merges, amt):
    queue.put(addToBuffer(merges, amt))
