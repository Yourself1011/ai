import csv
import math
import sys
import matplotlib.pyplot as plt

with open(sys.argv[1]) as f:
    f = csv.reader(f)
    iteration, loss = zip(*f)

iterations = list(map(int, iteration))
losses = list(map(lambda x: float(x), loss))
x = range(len(iterations))

fig, ax = plt.subplots()
ax.plot(x, losses)
fig.set_size_inches(16, 8, forward=True)
# plt.xticks(x, rotation=90)
plt.ylabel("loss")
plt.xlabel("iterations")
plt.show()
