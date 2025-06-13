import time
from multiprocessing import Process, Queue

from tokenizer import encode
from wikiPage import wikiPage

tokens = []
buffer = []
process = Process()
queue = Queue()
i = 0


def getData(amt: int, merges):
    global i, tokens, process, queue, buffer
    # print(i, len(tokens))
    if i + amt + 1 >= len(tokens):
        if len(tokens) == 0 and len(buffer) == 0:
            # first time running it
            addToBuffer(queue, merges, amt)
            buffer = queue.get()
        else:
            buffer = queue.get()
            process.join()
        chunk = tokens[i:] + buffer[: amt - (len(tokens) % amt) + 1]
        tokens = buffer[amt - (len(tokens) % amt) :]
        queue = Queue()
        process = Process(target=addToBuffer, args=[queue, merges, amt])
        process.start()
        i = 0
    else:
        chunk = tokens[i : i + amt + 1]
        i += amt
    return chunk


def addToBuffer(queue, merges, amt):
    buffer = []
    while len(buffer) < amt:
        start = time.time()
        filtered = wikiPage()
        new = encode(filtered, merges) + [256]
        buffer += new
        print(
            "    wiki article finished in",
            time.time() - start,
            "s,",
            len(new),
            "tokens",
        )
    queue.put(buffer)
