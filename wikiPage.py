from time import sleep
import regex
import requests


def wikiPage():
    err = True
    res = requests.Response()
    while err:
        try:
            res = requests.get("https://en.wikipedia.org/wiki/Special:Random")
            err = False
        except Exception as e:
            err = True
            print(e)
            sleep(10)
    text = res.text
    filtered = regex.sub(
        "[\n\t]+",
        "\n",
        regex.sub(
            r"(<style[\s\S]*?>[\s\S]*?</style>)|(<script[\s\S]*?>[\s\S]*?</script>)|(<[\s\S]*?>)|(<\/[\s\S]*?>)",
            "\n",
            text,
        ),
    )

    return filtered
