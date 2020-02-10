
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-1, 1, 0.1)
y1 = np.exp(x)
y2 = np.exp(2 * x)

plt.plot(x, y1, color="r", linestyle="-", marker=".", linewidth=1, label = 'y1')

plt.plot(x, y2, color="b", linestyle="-", marker=".", linewidth=1, label = 'y2')

plt.xlabel("x", fontsize = 11)
plt.ylabel("y")
plt.grid(color="k", linestyle=":")
plt.legend(loc='upper left')
plt.title("Figure 1")

plt.show()
