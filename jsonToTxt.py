import json

with open("data2.json", "r") as file:
    data = json.load(file)
    with open("data.txt", "a") as txt:
        for msg in data["messages"]:
            txt.write(
                f"{msg['author']['nickname']}:\n{msg['content']}<|endoftext|>\n\n"
            )
