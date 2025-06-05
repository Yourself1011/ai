from multiprocessing import Process
import time
from tokenizer import encode
from wikiPage import wikiPage

tokens = []
buffer = []
process = Process()
i = 0


def getData(amt: int, merges):
    global i, tokens, buffer, process
    if i >= len(tokens):
        i = 0
        if len(tokens) == 0 and len(buffer) == 0:
            # first time running it
            addToBuffer(merges)
        else:
            process.join()
        tokens = buffer
        process = Process(target=addToBuffer, args=[merges])
        process.start()

    chunk = tokens[i : min(len(tokens) - 1, i + amt)]
    i += amt
    return chunk


def addToBuffer(merges):
    global buffer
    print("new wiki article")
    start = time.time()
    filtered = wikiPage()
    buffer = encode(filtered, merges)
    print("wiki article finished in", time.time() - start, "s")
