import json
from os import error
import os
import time
from typing import Tuple
import regex as re

from wikiPage import wikiPage


specialTokens = {"<|endoftext|>": 256}
specialRegex = re.compile(
    "(" + ")|(".join([re.escape(x) for x in specialTokens.keys()]) + ")"
)


def tokenizer(
    text: str, size: int
) -> Tuple[dict[Tuple[int, int], int], dict[int, bytes]]:
    chunks = []

    for passage in re.split(specialRegex, text):
        if not re.match(specialRegex, passage):
            chunks += re.findall(
                r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",  # stolen straight from gpt-4, with some modification
                passage,
            )
        else:
            chunks += [passage]

    ids = [
        [x.encode("utf-8")] if re.match(specialRegex, x) else list(x.encode("utf-8"))
        for x in chunks
    ]
    unsaturated = set(range(len(ids)))
    merges: dict[Tuple[int, int], int] = {}
    vocab = {i: bytes([i]) for i in range(256)}
    for k, v in specialTokens.items():
        vocab[v] = k.encode("utf-8")
    initVocabLen = len(vocab)

    start = time.time()
    for i in range(size - initVocabLen):
        if not i % 100:
            print(f"{i}/{size - initVocabLen} {time.time() - start}s")
            start = time.time()

        # count pairs
        pairs = {}
        for c in unsaturated.copy():
            chunk = ids[c]
            if len(chunk) == 1:
                unsaturated.remove(c)
                continue
            for j in range(len(chunk) - 1):
                pairs[(chunk[j], chunk[j + 1])] = (
                    1
                    if (chunk[j], chunk[j + 1]) not in pairs
                    else pairs[(chunk[j], chunk[j + 1])] + 1
                )

        if len(pairs) == 0:
            print("no more pairs :skull:")
            break

        highestPair = max(pairs.keys(), key=lambda x: pairs[x])
        newId = len(vocab)
        merges[highestPair] = newId
        vocab[newId] = vocab[highestPair[0]] + vocab[highestPair[1]]

        for j in unsaturated:
            chunk = ids[j]
            k = 0
            newIds = []
            while k < len(chunk):
                if k != len(chunk) - 1 and (chunk[k], chunk[k + 1]) == highestPair:
                    newIds.append(newId)
                    k += 2
                else:
                    newIds.append(chunk[k])
                    k += 1
            ids[j] = newIds

    return (merges, vocab)


def encode(text: str, merges: dict[Tuple[int, int], int]) -> list[int]:
    chunks = []

    for passage in re.split(specialRegex, text):
        if not re.match(specialRegex, passage):
            chunks += re.findall(
                r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",  # stolen straight from gpt-4, with some modification
                passage,
            )
        else:
            chunks += [passage]

    ids: list[list[int]] = [
        [specialTokens[x]] if re.match(specialRegex, x) else list(x.encode("utf-8"))
        for x in chunks
    ]
    # print(chunks)
    # print(ids)
    reverse = {}
    for i in range(len(ids)):
        chunk = ids[i]
        for j in range(len(chunk)):
            c = chunk[j]
            if c in reverse:
                reverse[c].append((i, j))
            else:
                reverse[c] = [(i, j)]

    for pair, newId in merges.items():
        if pair[0] in reverse:
            new = []
            for i, char in reverse[pair[0]]:
                word = ids[i]
                if word[char] == -1:
                    continue

                j = char + 1
                while j < len(word) and word[j] == -1:
                    j += 1

                if j < len(word) and word[j] == pair[1]:
                    if newId not in reverse:
                        reverse[newId] = [(i, char)]
                    else:
                        reverse[newId].append((i, char))
                    word[char] = newId
                    word[j] = -1
                else:
                    new.append((i, char))
            reverse[pair[0]] = new

    out = []
    for chunk in ids:
        for token in chunk:
            if token != -1:
                out.append(token)
    return out

    for pair, newId in merges.items():
        for i in range(len(ids)):
            chunk = ids[i]
            j = 0
            newIds = []
            while j < len(chunk):
                if j != len(chunk) - 1 and (chunk[j], chunk[j + 1]) == pair:
                    newIds.append(newId)
                    j += 2
                else:
                    newIds.append(chunk[j])
                    j += 1
            ids[i] = newIds

    return sum(ids, [])


def decode(ids: list[int], vocab: dict[int, bytes]):
    return b"|".join([vocab[x] for x in ids]).decode("utf-8", errors="replace")


def load(vocabSize: int):
    merges = {}
    vocab = {}
    failed = False
    data = {}
    try:
        with open("data/tokenizerData.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        failed = True

    if not failed and "merges" in data and "vocab" in data:
        jsonMerges = data["merges"]
        for k, v in jsonMerges.items():
            merges[tuple(int(x) for x in k.split(","))] = v
        jsonVocab = data["vocab"]
        for k, v in jsonVocab.items():
            vocab[int(k)] = bytes(v)

    else:
        print("no previous tokenizer data found, making new one")
        data = ""
        for root, _, files in os.walk("data/training"):
            for name in files:
                with open(os.path.join(root, name), "r") as file:
                    data += file.read() + "\n"
        while len(data) < 10_000_000:
            data += wikiPage() + "\n"
            print(len(data))
        (merges, vocab) = tokenizer(data, vocabSize)

        with open("data/tokenizerData.json", "w") as file:
            jsonMerges = {}
            for k, v in merges.items():
                jsonMerges[",".join(map(str, k))] = v
            jsonVocab = {}
            for k, v in vocab.items():
                jsonVocab[k] = list(v)
            json.dump({"merges": jsonMerges, "vocab": jsonVocab}, file)

    return (merges, vocab)


if __name__ == "__main__":
    (merges, vocab) = load(50257)

    # print(merges)
    # print(vocab)
    # encoded = encode(
    #     "hello world donald duck jef codes le fang yin gary garf510 obama skibidi according to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. But the bee flies anyway, because bees don't care what humans think. Somebody once told me the world was gonna roll me, but I ain't the sharpest tool in the shed <|endoftext|>",
    #     merges,
    # )
    encoded = encode(
        input(),
        merges,
    )
    print(encoded)
    print(decode(encoded, vocab))
