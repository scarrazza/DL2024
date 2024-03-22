#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf


def true_fun(x):
    return np.cos(1.5 * np.pi * x)


def main():
    n_samples = 30
    np.random.seed(0)
    x = np.sort(np.random.rand(n_samples))
    y = true_fun(x) + np.random.randn(n_samples) * 0.1
    x_test = np.linspace(0, 1, 100)

    plt.figure()
    degrees = [1, 4, 15]

    def loss(p, func):
        ypred = func(list(p), x)
        return tf.reduce_mean(tf.square(ypred - y)).numpy()

    for degree in degrees:
        res = minimize(loss, np.zeros(degree+1), args=(tf.math.polyval), method='BFGS')
        plt.plot(x_test, np.poly1d(res.x)(x_test), label=f"Poly degree={degree}")

    plt.plot(x_test, true_fun(x_test), label="True function")
    plt.scatter(x, y, color='b', label="Samples")
    plt.title("TensorFlow")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0,1])
    plt.ylim([-2,2])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
