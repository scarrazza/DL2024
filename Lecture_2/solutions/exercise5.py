# import libraries
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return - np.sin(x * x) / x + 0.01 * x * x

# load data
x = np.linspace(-3, 3, 100)
y = f(x)
np.savetxt("output.dat", np.vstack([x,y]).T)

# plot
plt.plot(x, y, 'o-', label='$-\\frac{sin(x^2)}{x} + 0.01 * x^2$')
plt.title("")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim([-3,3])
plt.legend(frameon=False)

# save file to disk
plt.savefig("output5.png")

# open canvas
plt.show()
