#!/usr/bin/env python
import random


class Variable:
    def __init__(self, name):
        self.name = name

    def sample(self, size):
        raise NotImplementedError()


class Normal(Variable):
    def __init__(self, name):
        super().__init__(name)
        self.mu = 0
        self.sigma = 1

    def sample(self, size):
        """Performs normal sampling."""
        samples = []
        for _ in range(size):
            s = random.normalvariate(self.mu, self.sigma)
            samples.append(s)
        return samples


if __name__ == "__main__":
    n = Normal("mynormalinheritance")
    s = n.sample(100)
    print(s)
