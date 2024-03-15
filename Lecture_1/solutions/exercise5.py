#!/usr/bin/env python

def mean(x):
    return sum(x) / len(x)


def factorial(x):
    if x == 0:
        return 1
    else:
        return x * factorial(x - 1)


if __name__ == "__main__":
    v = [2, 6, 3, 8, 9, 11, -5]
    print(mean(v))
    print(factorial(5))
