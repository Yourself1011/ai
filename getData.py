from time import sleep, time
import regex
import requests
import urllib.parse
from datasets import load_dataset


def wikiPage():
    err = True
    res = requests.Response()
    while err:
        try:
            res = requests.get(
                "https://en.wikipedia.org/w/api.php?action=query&format=json&list=random&rnnamespace=0"
            )
            title = urllib.parse.quote_plus(
                res.json()["query"]["random"][0]["title"].replace(" ", "_")
            )
            url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles={title}&formatversion=2&explaintext=1"
            res = requests.get(url)
            print(f"got https://en.wikipedia.org/w/index.php?title={title}")
            err = False
        except Exception as e:
            err = True
            print("error fetching wiki article:")
            print(e)
            sleep(10)
    text = res.json()["query"]["pages"][0]["extract"]
    # filtered = regex.sub(
    #     "([\n\t]+ *)+",
    #     "\n",
    #     regex.sub(
    #         r"(<style[\s\S]*?>[\s\S]*?</style>)|(<script[\s\S]*?>[\s\S]*?</script>)|(<[\s\S]*?>)|(<\/[\s\S]*?>)",
    #         " ",
    #         text,
    #     ),
    # )
    # print(text)

    return text


dataset = None


def pj():
    global dataset
    if dataset is None:
        print("loading dataset")
        start = time()
        dataset = iter(load_dataset(
            "cerebras/SlimPajama-627B", split="train", streaming=True
        ).shuffle(seed=round(time() * 1000)))
        print(f"loaded dataset in {time() - start}s")

    return next(dataset)["text"]


if __name__ == "__main__":
    print(pj())
    print("\n")
    print(pj())
