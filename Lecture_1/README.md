# DL laboratory 1

### Prof. Stefano Carrazza

**Summary:** Modern applications of deep learning require the knowledge of
Python, a high level language. In this lecture we go step by step through the
basics of Python.

## Exercise 1 - Environment preparation

To log into the machines of the computational laboratory in the physics
department you must register through:

[https://studenti.fisica.unimi.it/registrazione](https://studenti.fisica.unimi.it/registrazione).

For this tutorials we have setup in the laboratory machines an anaconda
environment with all required packages.

In order to activate the environment open a terminal and do:

```bash
module load python3/anaconda
```

This will select a `python` 3.x interpreter from the [anaconda
distribution](https://www.anaconda.com/).

Then, to enable the tutorial environment with all packages relevant for DL do:
```bash
source activate DeepLearning
```

If you are working in our own machine we suggest to install
[anaconda3](https://www.anaconda.com/) and then perform the following
installations with:
```bash
conda install tensorflow keras numpy matplotlib pandas scikit-learn seaborn ipython-notebook numba
```

## Exercise 2 - Hello World!

- Write a python script which outputs the string `"Hello World!"` using the
  built-in python function `print`.

- Python is a dynamic typing language, allocate the following three variables
  `5`, `6.5` and `"hello"` using the function `input` and print to terminal
  using f-strings.

## Exercise 3 - Translating from C++

1. Translate the following C++ code in a python script:
    ```c++
    #include <iostream>

    int main()
    {
        int a = 1, b = 1;
        int target = 48;
        for(int n = 3; n <= target; ++n)
        {
            int fib = a + b;
            std::cout << "F("<< n << ") = " << fib << std::endl;
            a = b;
            b = fib;
        }

        return 0;
    }
    ```
2. Compare the output of the C++ code with the python code.


## Exercise 4 - Python basics

In this tutorial we will use `python` through `jupyter` notebooks.

In order to start a session open a terminal and do:
```bash
jupyter notebook
```
This command will open a browser with an interactive python3 session.

Commands are executed cell by cell by pressing SHIFT+ENTER.

1. Create an empty list `v` and append integers from [0,15] using a `for` loop.

3. Append 3 floats to an empty list.

4. Create a dictionary with the following keys `('name', 'loss', 'weights')`. Initialize the dictionary with the following default values `('neuralnet', 0.12, [10,25,5])`. Print to screen the value of `weights`. Assign to `loss` the value 10.

5. Given the list [2,6,3,8,9,11,-5] compute the mean value using a `for` loop.

6. Define a list using the expression `2 ** n` with `n = 10`.

## Exercise 5 - Functions

1. Write a function which returns the average from a a list. Test the function using the list `v=[2,6,3,8,9,11,-5]`.

2. Write the factorial function: `f(0) = 1`, `f(x) = x * f(x-1)`.

## Exercise 6 - Class 1

Given the base class:
```python
class Variable:
    def __init__(self, name):
        self.name = name

    def sample(self, size):
        raise NotImplementedError()
```

Implement an inherited `Normal` class which outputs a list of normal samples [mu=0, sigma=1] by overriding the `Variable.sample` method.

## Exercise 7 - Class 2

Construct a class which constructs and evaluates a 1D polynomial model with the following API:
- the class constructor must take the polynomial degree as argument.
- implement a `set_parameters` and `get_parameters` methods to update the parameter list.
- provide an `execute` method to access the polynomial prediction at a specific value in `x`.
