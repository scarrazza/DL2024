# DL laboratory 7

### Prof. Stefano Carrazza

**Summary:** CNNs, classification and localization.

## Exercise 1: A simple CNN classifier

1. Download the CIFAR10 dataset using `tensorflow.keras.datasets.cifar10.load_data()`. This dataset contains 60k (50k training / 10k test) low resolution color images for 10 classes: `['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`.

2. Verify data by plotting image samples.

3. Allocate a sequential model containing a normalization layer after determining the best choice for this particular dataset. After this initial normalization layer, threat images as a flatten layer, and build a classifier. Train the classifier using the test data as validation set. Store and plot the accuracy for the training and validation achieved with this approach.

4. Redesign the previous model by replacing the initial flatten layer with convolutional layers (`tf.keras.layers.Conv2D`) followed by a Max Pooling 2D layer (`tf.keras.layers.MaxPooling2D`). Try this model with 3 consecutive layers. Compare the results with the previous model.

## Exercise 2: Localization and classification

1. We provide images of 75x75 pixels containing MNIST digits in the `data.tgz` folder above. Each image contains only one digit of 28x28 pixels placed in a random position of the image. The files `training_images.npy` and `validation_images.npy` contains the images, `training_labels.npy` and `validation_labels.npy` the labels of each image, `training_boxes.npy` and `validation_boxes.npy` the 4 coordinates of the bounding boxes (xmin, ymin, xmax, ymax). Load data and plot samples.

2. Construct a custom Keras model (using the functional API) with the following components: a feature extractor using a CNN followed by a flatten and a dense layer and two final end-points: a classifier (10 classes) and a bounding box regressor (4 coordinates). Use the categorical cross-entropy loss function for the classifier and the MSE for the bounding box regressor.

3. Plot the classification and bounding box losses. Verify the results on the validation dataset by plotting samples and computing the IoU. Evaluate the total number of good and bad bounding box predictions using an IoU threshold of 0.6.
