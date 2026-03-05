import csv
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep

fig, ax = plt.subplots()
fig.set_size_inches(16, 8, forward=True)
# plt.xticks(x, rotation=90)
plt.ylabel("loss")
plt.xlabel("iterations")


def update(frame):
    with open(sys.argv[1]) as f:
        f = csv.reader(f)
        iteration, loss = zip(*f)

    iterations = list(map(int, iteration))
    losses = list(map(lambda x: float(x), loss))
    x = range(len(iterations))

    ax.clear()
    (line,) = ax.plot(x, losses)

    print("whee")
    return [line]


update(None)
anim = animation.FuncAnimation(fig, update, interval=30000, cache_frame_data=False)
plt.show()
