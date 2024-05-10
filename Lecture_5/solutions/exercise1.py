#!/usr/bin/env python
import hyperopt
from hyperopt import hp, tpe, Trials, fmin, rand
import numpy as np
import matplotlib.pyplot as plt


# Point 1
def f(x):
    p = np.poly1d([1, -2, -28, 28, 12, -26, 100])
    return p(x) * 0.05


def main():
    space = hp.uniform('x', -5, 6)
    plt.hist([hyperopt.pyll.stochastic.sample(space) for _ in range(1000)])
    plt.title("Domain Space")
    plt.xlabel("x")
    plt.ylabel("Frequency")
    plt.show()

    # Run 2000 evals with the tpe algorithm
    tpe_trials = Trials()
    tpe_best = fmin(fn=f, space=space,
                    algo=tpe.suggest, trials=tpe_trials,
                    max_evals=2000)
    print(tpe_best)

    # Point 2-3
    x = np.linspace(-5, 6, 100)
    plt.plot(x, f(x))
    plt.axvline(tpe_best['x'], color='red')
    plt.title("Objective function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    # Point 4
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(tpe_trials.idxs_vals[0]['x'], tpe_trials.idxs_vals[1]['x'])
    ax1.axhline(tpe_best['x'], color='red')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("x value")
    ax1.set_title("TPE")
    ax2.hist(tpe_trials.idxs_vals[1]['x'], bins=100)
    ax2.set_xlabel("x value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("TPE")

    # Point 5
    rdn_trials = Trials()
    tpe_best = fmin(fn=f, space=space,
                    algo=rand.suggest, trials=rdn_trials,
                    max_evals=2000)
    print(tpe_best)

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(rdn_trials.idxs_vals[0]['x'], rdn_trials.idxs_vals[1]['x'])
    ax1.axhline(tpe_best['x'], color='red')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("x value")
    ax1.set_title("Random")
    ax2.hist(rdn_trials.idxs_vals[1]['x'], bins=100)
    ax2.set_xlabel("x value")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Random")
    plt.show()


if __name__ == "__main__":
    main()
