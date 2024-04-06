# DL laboratory 3

### Prof. Stefano Carrazza

**Summary:** TensorFlow and Keras basics.

## Exercise 1 - MLP network

Using TensorFlow primitives perform the following steps:

1. allocate random normal variables for weight and bias representation of a
  multi-layer perceptron (MLP) with `n_input` size, two hidden layers with
  `n_hidden_1` and `n_hidden_2` neurons respectively and `n_output` size.

2. define a function which takes a tensor as input and returns the MLP
  prediction. Use the sigmoid function as activation function for all nodes in
  the network except for the output layer, which should be linear.

3. Test the model prediction for 10 values in x linearly spaced from [-1,1] with
  `n_input=1`, `n_hidden_1=5`, `n_hidden_2=2`, `n_output=1`.

## Exercise 2 - Sequential model

- Translate the previous exercise with TensorFlow/Keras's sequential model.
- Print the model summary to screen.
- Verify that predictions between both models are in agreement.
- Print the weights from the model object.

## Exercise 3 - Manual training with functional API

**Data generation**

1. Generate predictions of `f(x) = 3 * x + 2` for 200 linearly spaced `x` points
   between [-2, 2] in single precision.

2. Include random normal noise (mu=0, sigma=1) to all predictions.

3. Plot data and ground truth model.

**Linear fit**

4. Define a custom model using `tf.Module` inheritance which returns the
   functional form `w * x + b` where `w` and `b` are tensor variables
   initialized with random values.

5. Define a loss function matching the mean squared error.

6. Plot data, ground truth model, predictions and loss function for the
   untrained model.

**Training loop**

7. Define a `train` function which computes the loss function gradient and
   performs a full batch SGD (manually).

8. Define a `training_loop` function which takes performs 10 epochs, prints the
   loss function at each iteration to screen and stores the model weights.

**Post-fit**

9. Print the evolution of weights at each iteration.

10. Plot data, ground truth model, predictions and loss function after the training.

**Use Keras**

11. Replace the training loop with Keras model API, check results.
