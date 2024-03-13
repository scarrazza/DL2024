# DL laboratory 2

### Prof. Stefano Carrazza

**Summary:** In this lecture we go step by step through the basics of Python,
Numpy, Matplotlib and TensorFlow.

## Exercise 1 - Numpy basics

- Allocate an array `a` from the python list `[[0.5, -1],[-1, 2]]` with dtype float 32bit.

- Verify its shape, dimension and create a deep copy `b` using a flatten shape.

- Assign zero to elements in `b` with even indexes.

## Exercise 2 - Numpy performance

Write a code which starting from a specific space dimension `N` allocates:
- a random vector `v` of real double-precision (`numpy.float64`) of size `N`.
- a random square matrix `A` with size `NxN`.
- implement a function which performs the dot product using only python primitives.
- measure the execution time of the previous function and compare with NumPy's `dot` method.
- accelerate the python dot product using the [Numba](https://numba.pydata.org/) library.
- compare the performance results.

## Exercise 3 - Matplotlib basics

Write a plotting script (or notebook) using
[Matplotlib](https://matplotlib.org/) for the function `exp(-x) * cos(2*pi*x)`
for 100 points in x linearly spaced from 0 to 5.

## Esercizio 4 - Scatter plot

- Download the `data4.dat` file using:
    ```
    wget https://raw.githubusercontent.com/scarrazza/DL2024/main/Lecture_2/data4.dat
    ```
  The file contains 2 columns (x,y) of points.

- Load data using `numpy` and use `matplotlib` scatter plot for the graphical representation.

- Update title with "Charged particles", axis titles with "x-coordinate" e "y-coordinate".

- Color points with red.

- Store plot to disk using the filename `output.png`.

## Esercizio 5 - Plot di funzioni

Write a python script/notebook with the following steps:

- Define a function `f(x) = -sin(x*x)/x + 0.01 * x*x`.

- Generate an array with 100 elements, linearly spaced between [-3, 3].

- Write to a file `output.dat` the values of `x` and `f(x)` line by line.

- Plot all points.

- Bound the x-axis between x=`[-3,3]`.

- Add title, axis labels and a line between points, show the equation in the legend.

- Store plot to disk as `output5.png`.

## Exercise 6 - Pandas basics

1. Download and import the following dataset using pandas:
    ```python
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                            na_values='?', comment='\t',
                            sep=' ', skipinitialspace=True)
    ```

2. Print the mean values of each column.

3. Filter results by selecting only entries where the number of cylinders is
  equal to 3.

## Exercise 7 - Polynomial fits with Numpy

- Define a true function `true_function(x) = cos(1.5 * pi * x)`.
- Generate 30 random points in x between [0, 1].
- Evaluate the target points as: `true_function(x) + np.random.rand() * 0.1`.
- Implement and perform polynomial fits for degrees 1, 4 and 15. Use as loss
  function the MSE function.

## Exercise 8 - Polynomial fits with TensorFlow

Repeat the previous exercise by replacing the Numpy primitives with TensorFlow primitives.
