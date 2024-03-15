#!/usr/bin/env python

class Polynomial:
    def __init__(self, degree):
        self.degree = degree
        # growing ordering a + b * x + c * x**2 ...
        self._parameters = [0.0] * (degree + 1)

    def set_parameters(self, parameters):
        self._parameters = parameters

    def get_parameters(self):
        return self._parameters

    def _execute(self, x):
        r = 0
        for d in range(self.degree):
            r += self._parameters[d] * x ** d
        return r

    def __call__(self, x):
        return self._execute(x)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, p):
        if len(p) != self.degree + 1:
            raise ValueError("Parameter lenght does not match polynomial degree.")
        self._parameters = p


if __name__ == "__main__":

    p = Polynomial(2)
    p.set_parameters([1, 2, 3])
    print(f"params = {p.get_parameters()}")
    print(f"pol = {p(2)}")

    # now with setter and getter
    p.parameters = [3, 2, 1]
    print(f"params = {p.parameters}")
    try:
        p.parameters = [1]
    except ValueError:
        print("Error intercepted")
    except:
        raise RuntimeError("Error not expected")

