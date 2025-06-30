import csv
import matplotlib.pyplot as plt

with open("data/history.csv") as f:
    f = csv.reader(f)
    iteration, loss = zip(*f)

iterations = list(map(int, iteration))
losses = list(map(float, loss))
x = range(len(iterations))

plt.plot(x, losses)
plt.xticks(x, rotation=90)
plt.ylabel("loss")
plt.show()
