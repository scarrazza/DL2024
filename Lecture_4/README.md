# DL laboratory 4

### Prof. Stefano Carrazza

**Summary:** Regression and classification.

## Exercise 1 - Regression with sequential model

Write a ML regression model using TensorFlow/Keras's sequential model with the following steps:

**Data Loading**

1. Download the data with:
    ```
    wget https://raw.githubusercontent.com/scarrazza/DL2024/main/Lecture_4/data.dat
    # or (on OSX)
    curl https://raw.githubusercontent.com/scarrazza/DL2024/main/Lecture_4/data.dat -o data.dat
    ```
    This file contains an undefined number of data points already divided into training and validation (x_tr, y_tr, x_val, y_val). The data is 2D dimensional.

2. Load and plot data for training and validation.

**Baseline linear fit**

3. Create a baseline linear model (dense layer with 1 unit node) and store the instance of this class in a variable called `model`.

4. Compile `model` using:
    - the mean squared error as loss function,
    - the `tensorflow.keras.optimizers.SGD` class with learning rate 0.01 as optimizer.

5. Perform a fit with `model.fit` with full batch size and 500 epochs. Monitor the validation data during epochs.

6. Plot the loss function for training and validation using the history object returned by `model.fit`.

7. Plot the model prediction on top of data.

**NN model fit**

8. Build a neural network model with 3 layers containing 10 nodes each and `relu` activation function and a last layer with a single unit and linear activation function.

9. Perform a fit using the same setup in 4-6.

10. Plot the loss function history for training and validation.

11. Plot the model prediction on top of data.


## Exercise 2 - Classification with sequential model

Write a ML classification model using keras with the following steps:

**Data loading**

1. Load the fashion mnist dataset from `tensorflow.keras.datasets.fashion_mnist`. Study the dataset size (pixel shape) and plot some sample images. This dataset contains the following classes `['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']`.

2. Normalize images for training and test, considering the maximum pixel value of 255.

**NN model fit**

3. Build a NN model with flattens images and applies 2 dense layers with 128 and 10 units respectively. The first layer uses `relu` while the last layer `softmax`. Determine the number of trainable parameters.

4. Fit the dataset with 5 epochs, using `adam`'s optimizer, and the `sparse_categorical_crossentropy` loss function. The `Sequential.compile` method supports extra arguments, such as `metrics=['accuracy']` in order to monitor extra statistical estimators during epochs.

5. Evaluate test accuracy.

6. Identify examples of bad classification.
