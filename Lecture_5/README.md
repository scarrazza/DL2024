# DL laboratory 5

### Prof. Stefano Carrazza

**Summary:** Hyperparameter tune.

## Exercise 1: Bayesian optimization

1. Define an objective function which returns the analytic expression of a 1d
   polynomial with coefficients `f(x) = 0.05 (x^6 - 2 x^5 - 28 x^4 + 28 x^3 + 12 x^2 -26 x + 100)`.

2. Plot the previous function using a linear grid of points in x between [-5, 6].

3. Define an uniform search domain space using hyperopt. With
   `hyperopt.pyll.stochastic.sample` build an histogram with samples from that
   space.

4. Perform the objective function minimization using the Tree-structured Parzen
   Estimator model, 2000 evaluations and store the trials using
   `hyperopt.Trials`. Print to screen the best value of x. Show scatter plot
   with the x-value vs iteration together with the final best value of x. Show
   the histogram of x-values extracted during the scan.

5. Repeat the previous point now using a random search algorithm.

## Exercise 2: Hyperparameter scan for classifier

Write a DL regression model using Keras with the following steps:

**Data loading**

1. Load the mnist dataset from `tensorflow.keras.datasets.mnist`. Study the dataset size (shape) and normalize the pixels.

**DNN model**

2. Design a NN architecture for the classification of all digits.

**Hyperparameter scan**

3. Define a function which parametrizes the learning rate and the number of
   units of the DNN model using a python dict.

4. Use the Tree-structured Parzen Estimator with the
   [hyperopt](http://hyperopt.github.io/hyperopt/) library.

5. Plot the accuracy vs learning rate and number of layers for each trial.
