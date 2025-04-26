import json
import os

for root, _, files in os.walk("data/discordJson/toProcess"):
    for name in files:
        filePath = os.path.splitext(name)[0]
        with open(os.path.join("data/discordJson/toProcess", name), "r") as file:
            data = json.load(file)
            with open(os.path.join("data/training", filePath + ".txt"), "a") as txt:
                for msg in data["messages"]:
                    txt.write(f"{msg['author']['nickname']}:\n{msg['content']}\n\n")
