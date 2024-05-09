# DL laboratory 6

### Prof. Stefano Carrazza

**Summary:** Callbacks and RNNs.

## Exercise 1: Early stopping

1. Download the IRIS dataset using the method `sklearn.dataset.load_iris`. The
   `load_iris` method returns a dictionary with the following keys: `data` which
   contains a matrix with the sepal and petal lengths and widths,
   `feature_names` with the corresponding list of strings for the data names, `target` the corresponding flower label id, `target_names` the list of strings for the flower names.

2. Load data in a pandas DataFrame. This should include the `data` values and
   the flower `label`. Print the DataFrame to screen. *(Optional)* Inspect the data with e.g. correlation/scatter plots.

3. Given that the `label` column is a categorical feature, we should convert it
   to a one-hot encoding. Perform this one-hot encoding operation by replacing the `label` column with three new columns: `label_setosa`, `label_versicolor`, `label_virginica`.

4. Extract 80% of the data for training and keep 20% for test, using the
   `DataFrame.sample` method.

5. Define a sequential model with 3 output nodes with `softmax` activation
   function. Perform a fit using Adam and the categorical cross-entropy loss
   function for 200 epochs, validation split of 40% and batch size of 32. Plot the learning curves (loss vs epochs) for the training and validation datasets.

6. Modify the previous point in order to use early stopping on the validation
   loss with patience=10. Plot the learning curves and check the stopping epoch.

7. *(Optional)* Include the TensorBoard callback. Integrate the hyperopt
   pipeline implemented in the previous lecture using a `loss` the accuracy on
   the test set obtained in point 4.

## Exercise 2: Forecasting time series

1. We provide numpy arrays for daily measurements performed in blocks of 10
   days. The data was already filtered and normalized. Download the following
   datasets and check the corresponding sizes:
    ```bash
    # for the training set
    wget https://raw.githubusercontent.com/scarrazza/DL2024/main/Lecture_6/training_data.npy
    wget https://raw.githubusercontent.com/scarrazza/DL2024/main/Lecture_6/training_label.npy
    # for the test set
    wget https://raw.githubusercontent.com/scarrazza/DL2024/main/Lecture_6/test_data.npy
    wget https://raw.githubusercontent.com/scarrazza/DL2024/main/Lecture_6/test_label.npy
    ```

2. Build and train an LSTM model using Adam with MSE loss, 25 epochs, batch size
   32. Print the final MSE for the test set.

3. Plot the following quantities: training and test data vs days, the LSTM
   predictions for the test data, the LSTM predictions for the first 100 days,
   the residual (y_test - prediction), and the scatter plot between the true
   test data vs predictions.
