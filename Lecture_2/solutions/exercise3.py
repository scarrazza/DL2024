#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


if __name__ == "__main__":
    t1 = np.linspace(0.0, 5.0, 100)
    plt.plot(t1, f(t1), 'bo')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('example')
    plt.show()
