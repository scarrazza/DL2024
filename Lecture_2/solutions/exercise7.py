#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize


def true_fun(x):
    return np.cos(1.5 * np.pi * x)


def main():
    n_samples = 30
    np.random.seed(0)
    x = np.sort(np.random.rand(n_samples))
    y = true_fun(x) + np.random.randn(n_samples) * 0.1
    x_test = np.linspace(0, 1, 100)

    plt.figure(figsize=(14, 5))
    degrees = [1, 4, 15]
    ax = plt.subplot(1, len(degrees), 1)

    # Mode 1 - using least squares
    for degree in degrees:
        p = np.polyfit(x, y, degree)
        z = np.poly1d(p)
        plt.plot(x_test, z(x_test), label=f"Poly degree={degree}")
    plt.plot(x_test, true_fun(x_test), label="True function")
    plt.scatter(x, y, color='b', label="Samples")
    plt.title("Polyfit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0,1])
    plt.ylim([-2,2])
    plt.legend()

    # Mode 2 - curve fitting
    ax = plt.subplot(1, len(degrees), 2)

    def poly1(x, a, b):
        return  a * x + b
    def poly4(x, a, b, c, d, e):
        return  a * x**4 + b * x**3 + c * x**2 + d * x + e
    def poly15(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15):
        return  a15*x**15+a14*x**14+a13*x**13+a12*x**12+a11*x**11+a10*x**10+a9*x**9+a8*x**8 + \
                a7*x**7+a6*x**6+a5*x**5+a4*x**4+a3*x**3+a2*x**2+a1*x+a0

    popt, pcov = curve_fit(poly1, x, y)
    plt.plot(x_test, poly1(x_test, *popt), label="Poly degree=1")
    popt, pcov = curve_fit(poly4, x, y)
    plt.plot(x_test, poly4(x_test, *popt), label="Poly degree=4")
    popt, pcov = curve_fit(poly15, x, y)
    plt.plot(x_test, poly15(x_test, *popt), label="Poly degree=15")

    plt.plot(x_test, true_fun(x_test), label="True function")
    plt.scatter(x, y, color='b', label="Samples")
    plt.title("Scipy.curve_fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0,1])
    plt.ylim([-2,2])
    plt.legend()

    # Mode 3 - scipy minimize
    ax = plt.subplot(1, len(degrees), 3)

    def loss(p, func):
        ypred = func(p)
        return np.mean(np.square(ypred(x) - y))

    for degree in degrees:
        res = minimize(loss, np.zeros(degree+1), args=(np.poly1d), method='BFGS')
        plt.plot(x_test, np.poly1d(res.x)(x_test), label=f"Poly degree={degree}")

    plt.plot(x_test, true_fun(x_test), label="True function")
    plt.scatter(x, y, color='b', label="Samples")
    plt.title("Scipy.minimize")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0,1])
    plt.ylim([-2,2])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
